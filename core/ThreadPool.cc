#include <chrono>
#include <mutex>
#include <thread>
#include <iostream>
#include <vector>
#include <map>
#include <cstdio>
#include <unistd.h>
#include <functional>
#include "TransformationBase.hh"
#include "GNAObject.hh"
#include "ThreadPool.hh"
#include <algorithm>
#include <numeric>
#include <future>

void MultiThreading::Task::run_task() {
    if (m_entry->tainted){// && !m_entry->frozen) {
	m_entry->evaluate();
	std::cout << "EVA-id = " << std::this_thread::get_id() << std::endl;
        m_entry->tainted = false;
    }
}


MultiThreading::Worker::Worker(ThreadPool &in_pool) : pool(in_pool) {
    task_stack = new std::stack<Task>();
}


void MultiThreading::Worker::work(){ // runs task stack
 //   std::cout << "EVA-id-work() = " << std::this_thread::get_id() << std::endl;
  //  std::cerr << "Work" << std::endl;
    Task current_task;
    //std::lock_guard<std::mutex> lock(pool.tp_pop_mutex);
  //  std::cerr << "Work-after-locker" << std::endl;

    while (!task_stack->empty()) {
    	{
	    std::lock_guard<std::mutex> lock(pool.tp_pop_mutex);
    	    std::cout << "Work task size  = " << task_stack->size() << std::endl;
    	    current_task = task_stack->top();
//        if (!current_task.ready()) { /* make waiting for finish children */ }
      	    task_stack->pop(); 
	}
      	current_task.run_task(); // TODO: make it in separate thread
    }
    // TODO sleep thread

}


MultiThreading::ThreadPool::ThreadPool (int maxthr) : m_max_thread_number(maxthr) {
    // Mother thread creation
    std::cout << "Thread pool created" << std::endl;
    if (m_max_thread_number <= 0) m_max_thread_number = std::thread::hardware_concurrency();
    m_workers.push_back(Worker(*this));
    m_workers[0].thr_head = std::this_thread::get_id();
}


int MultiThreading::ThreadPool::whoami() {
        size_t thr_size = m_workers.size();
        for (size_t i =0; i < thr_size; i++) {
            if (m_workers[i].thr_head == std::this_thread::get_id())
                return i;
        }
        return -1;
    }


void MultiThreading::ThreadPool::new_worker(MultiThreading::Task &in_task, size_t index) {
        {	
	std::cout << "new_worker-index = " << index << std::endl;
		//std::lock_guard<std::mutex> lock(tp_add_mutex);
		m_workers.push_back(Worker(*this));
/*		{
			std::lock_guard<std::mutex> lock(tp_waitlist_mutex);
			m_global_wait_list.push_back({});
		}
*/
        	std::cerr << "New worker added! mw size = " << m_workers.size() << std::endl;
        	m_workers[index].thr_head =  std::this_thread::get_id();
		std::cout << "New worker ID = " << std::this_thread::get_id() << std::endl;
        	//if (in_task.done()) std::async(std::launch::async, std::bind(&Worker::work, m_workers[index]));
	}
	tp_add_mutex.unlock();
	cv_tp_add_mutex.notify_one();
        if (in_task.done()) m_workers[index].work();
}

void MultiThreading::ThreadPool::manage_not_motherthread(MultiThreading::Task in_task) {
    size_t src_size = in_task.m_entry->sources.size();
    if (src_size > 0) {
      for (size_t i = 1; i < src_size; i++) {                 // Try to make new thread                
            std::cout << "SECOND SOURCE! " << i << " size " << src_size << std::endl;
            if ( in_task.m_entry->sources[i].sink->entry->tainted) {
                add_task(Task(in_task.m_entry->sources[i].sink->entry));
            }
      }
    }
}

size_t MultiThreading::ThreadPool::try_to_find_worker(MultiThreading::Task in_task) {
    int worker_index = -1;

    std::cout << "WHOAMI = " << whoami() << std::endl;
    size_t w_size;
    {
        std::lock_guard<std::mutex> lock(tp_waitlist_mutex);
        w_size = m_workers.size();
    }

    auto curr_id = std::this_thread::get_id();

    for (size_t i = 0; i < w_size; i++)  {
      std::lock_guard<std::mutex> lock(tp_add_mutex);
      if((m_workers[i].is_free()) && worker_index == -1) {
        // || (!m_workers[i].is_free() && curr_id == m_workers[i].thr_head)) { 
        //std::cerr << "Free worker found!" << std::endl;
        {
                m_workers[i].add_to_task_stack(in_task);
                m_workers[i].thr_head = curr_id;
                worker_index = i;
                std::cout << "Fount worker index = " << worker_index << std::endl;
        }
      }
    }


    if (worker_index == -1) {
      if (w_size  < m_max_thread_number) {
        std::lock_guard<std::mutex> lock(tp_thr_add_mutex);
        //std::lock_guard<std::mutex> lock2(tp_add_mutex);
	std::unique_lock<std::mutex> lck(tp_add_mutex);
        threads.push_back(std::thread(
                [&in_task, w_size, this]() 
                {MultiThreading::ThreadPool::new_worker (std::ref(in_task), w_size - 1); 
                } ));
	cv_tp_add_mutex.wait(lck);
	std::cout << "Waiting-end!!!" << std::endl;
//	std::cout << "thr creat, id = " << std::this_thread::get_id() << "!!! mw size = " << m_workers.size() << std::endl;
//	m_workers.push_back(Worker(*this)); // HERE
        worker_index = threads.size()-1;
      } else {
        for (size_t i = 0; i < w_size; i++) {
          std::lock_guard<std::mutex> lock(tp_waitlist_mutex);
          std::cout << i << " " << m_workers[i].thr_head << " (try to add task to waitlist) ";
          if (m_workers[i].thr_head == curr_id && !in_task.done()) {
            std::cout << "Task added to wait list, m_global_wait_list size = "
                        << m_global_wait_list.size() << " i = " << i  << std::endl;
            m_global_wait_list[i].push_back(in_task);
          }
        }
      }
    }
    std::cout << "try-to-find worker_index = " << worker_index << " size =" <<  threads.size() << ";";
    return worker_index;

}


void MultiThreading::ThreadPool::add_task(MultiThreading::Task in_task) {

    manage_not_motherthread(in_task); 

    //std::cout << "src_size = " << src_size << std::endl;
    if (in_task.m_entry->sources.size() > 0) {
      if ( in_task.m_entry->sources[0].sink->entry->tainted) {
        Task motherthread_task(in_task.m_entry->sources[0].sink->entry);                                 
   	 
    //    add_task(motherthread_task);                                                               
      }
    }
   size_t worker_index = try_to_find_worker(in_task);
   std::cout << "worker_index = " << worker_index << "m_workers size = " << m_workers.size() << std::endl;
   if (in_task.done() && worker_index >= 0) { std::cout << "done -- now work "; m_workers[worker_index].work(); }
   else { std::cout << "eval only "; in_task.run_task(); } 

}

int MultiThreading::ThreadPool::is_free_worker_exists() {
    return 0;
    // returns index of free worker
    // if there is no free workers, returns -1
} 

bool MultiThreading::ThreadPool::is_pool_full () {
    // returns true if size of pool >= max pool size (usually setted externally)
    return (m_workers.size() >= m_max_thread_number);
}

