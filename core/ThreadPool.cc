#include <thread>
#include <iostream>
#include <vector>
#include <map>
#include <cstdio>
#include <unistd.h>
#include <functional>
#include "TransformationEntry.hh"
#include "GNAObject.hh"
#include "ThreadPool.hh"
#include <algorithm>
#include <numeric>
#include <future>

//struct Entry;

bool MultiThreading::Task::done() { 
	std::cerr << "DONE FUNC src size " << m_entry->sources.size() << std::endl; 
	return (!(m_entry->tainted) || (m_entry->sources.size() == 0)); 
}


void MultiThreading::Task::run_task() {
    std::cerr << "runtask-before-lock-taskmtx " <<std::endl;
    std::unique_lock<std::mutex> t_lck(task_mtx);
    task_cv.wait(t_lck, [this]{return !m_entry->running;});
    std::cerr << "runtask-after-lock-taskmtx " << std::endl;
    if (m_entry->tainted) {// && !m_entry->frozen) {
        //t_lck.lock();
        //task_cv.wait(t_lck);
        m_entry->mark_running();
std::cerr << "EVA-after-wait" << std::endl;
        if (m_entry->tainted){ 
	    m_entry->evaluate();
	    std::cerr << "EVA-id = " << std::this_thread::get_id() << std::endl;
            m_entry->tainted = false;
	}
	std::cerr << "RUNNING STAT = " << m_entry->running  << std::endl;
        m_entry->mark_not_running();
//	task_cv.notify_one();
    }
        //t_lck.unlock();
std::cerr << "EVA-after-unlock" << std::endl;
//        t_lck.unlock();
    task_cv.notify_one();
   
}


MultiThreading::Worker::Worker(ThreadPool &in_pool) : pool(in_pool) {
    task_stack = new std::stack<Task>();
}


void MultiThreading::Worker::work() { // runs task stack
    Task current_task;
//    std::cerr << "inside-work" << std::endl;
//	std::cerr << "work-before-lock-worker ";
 //   std::unique_lock<std::mutex> w_lck(mtx_worker);
//	std::cerr << "work-after-lock-worker ";
   // while (task_stack->empty()) {
//	std::cerr << "while-task-empty" <<std::endl;
   //     cv_worker.wait(w_lck);
  //      while (!task_stack->empty()) {
    	    
	        std::lock_guard<std::mutex> task_stack_lock(mtx_worker);
    	        current_task = task_stack->top();
      	        task_stack->pop(); 
 	   
         mtx_worker.unlock();
     	 current_task.run_task(); // TODO: make it in separate thread
     //    }
    //     w_lck.unlock();
//	 cv_worker.notify_all();
    //}
}


MultiThreading::ThreadPool::ThreadPool (int maxthr) : m_max_thread_number(maxthr) {
    // Mother thread creation
    std::cerr << "Thread pool created" << std::endl;
    if (m_max_thread_number <= 0) m_max_thread_number = std::thread::hardware_concurrency();
    m_workers.push_back(Worker(*this));
    m_workers[0].thr_head = std::this_thread::get_id();
}


int MultiThreading::ThreadPool::whoami() { 
	std::cerr << "in-whoami" << std::endl;
        std::lock_guard<std::mutex> lock(tp_m_workers_mutex);
        size_t thr_size = m_workers.size();
        for (size_t i =0; i < thr_size; i++) {
            if (m_workers[i].thr_head == std::this_thread::get_id())
                return i;
        }
        return -1;
    }


void MultiThreading::ThreadPool::new_worker(MultiThreading::Task &in_task, size_t index) {
//        {	
	std::unique_lock<std::mutex> lock(tp_m_workers_mutex);
		m_workers.push_back(Worker(*this));
/*		{
			std::lock_guard<std::mutex> lock(tp_waitlist_mutex);
			m_global_wait_list.push_back({});
		}
*/
        	m_workers[index].thr_head =  std::this_thread::get_id();
		std::cerr << "New worker ID = " << std::this_thread::get_id() << std::endl;
//	}
	tp_m_workers_mutex.unlock();
	cv_tp_m_workers_mutex.notify_one();
        if (in_task.done()) m_workers[whoami()].work();
	else in_task.run_task();
}

void MultiThreading::ThreadPool::manage_not_motherthread(MultiThreading::Task in_task) {
    size_t src_size = in_task.m_entry->sources.size();
    if (src_size > 0) {
      for (size_t i = 1; i < src_size; i++) {                 // Try to make new thread                
            std::cerr << "SECOND SOURCE! " << i << " size " << src_size << std::endl;
            if ( in_task.m_entry->sources[i].sink->entry->tainted) {
                add_task(Task(in_task.m_entry->sources[i].sink->entry));
            }
      }
    }
}

size_t MultiThreading::ThreadPool::try_to_find_worker(MultiThreading::Task in_task) {
    int worker_index = -1;

    size_t w_size;
    {
std::cerr << "try-to-lock-before-wait" << std::endl;
        std::lock_guard<std::mutex> lock(tp_m_workers_mutex);
std::cerr << "try-to-lock-afrer-wait" << std::endl;
        w_size = m_workers.size();
    }

    auto curr_id = std::this_thread::get_id();

    for (size_t i = 0; i < w_size; i++)  {
	std::cerr << "try-to-before-lock-add " << std::endl;
      std::lock_guard<std::mutex> lock(tp_m_workers_mutex);
	std::cerr << "try-to-after-lock-add " << std::endl;
      if((m_workers[i].is_free()) && worker_index == -1) {
        // || (!m_workers[i].is_free() && curr_id == m_workers[i].thr_head)) { 
        {
                m_workers[i].add_to_task_stack(in_task);
                m_workers[i].thr_head = curr_id;
                worker_index = i;
		m_workers[i].cv_worker.notify_one();
                std::cerr << "Fount worker index = " << worker_index << std::endl;
        }
	//lock.unlock();
      }
    }
std::cerr << "DEBUG TMP " << worker_index << std::endl;

    if (worker_index == -1) {
      if (w_size  < m_max_thread_number) {
	//std::cerr << "try-to-before-lock-thradd ";
	std::cerr << "THREADS!! " << std::endl;
        std::lock_guard<std::mutex> lock(tp_threads_mutex);
        //std::lock_guard<std::mutex> lock2(tp_add_mutex);
	//std::cerr << "try-to-before-lock-add ";
	//std::unique_lock<std::mutex> lck(tp_d_mutex);
	//std::cerr << "try-to-after-lock-add ";
//	cv_tp_threads_mutex.wait(lock);
        threads.push_back(std::thread(
                [&in_task, w_size, this]() 
                {MultiThreading::ThreadPool::new_worker (std::ref(in_task), w_size - 1); 
                } ));
	cv_tp_threads_mutex.notify_all();
	std::cerr << "Waiting-end!!!" << std::endl;
        worker_index = threads.size()-1;
      } else {
        for (size_t i = 0; i < w_size; i++) {
	std::cerr << "try-to-before-lock-wait " << std::endl;
          std::lock_guard<std::mutex> lock(tp_waitlist_mutex);
	std::cerr << "try-to-after-lock-wait " <<std::endl;
          std::cerr << i << " " << m_workers[i].thr_head << " (try to add task to waitlist) " << std::endl;
          if (m_workers[i].thr_head == curr_id && !in_task.done()) {
            std::cerr << "Task added to wait list, m_global_wait_list size = "
                        << m_global_wait_list.size() << " i = " << i  << std::endl;
            m_global_wait_list[i].push_back(in_task);
          }
        }
      }
    }
    std::cerr << "try-to-find worker_index = " << worker_index << " size =" <<  threads.size() << " " << std::endl;
    return worker_index;
}


void MultiThreading::ThreadPool::add_task(MultiThreading::Task in_task) {

std::cerr << "inside-add-task" << std::endl;
std::cerr << "id = " << std::this_thread::get_id() << std::endl;
    manage_not_motherthread(in_task); 
std::cerr << std::endl << "inside-add-task-afternotMT" << std::endl;

/*    if (in_task.m_entry->sources.size() > 0) {
      if ( in_task.m_entry->sources[0].sink->entry->tainted) {
        Task motherthread_task(std::ref(in_task.m_entry->sources[0].sink->entry));                                 
   	 
	std::cerr << "inside-add-taskIF" << std::endl;
        add_task(motherthread_task);                                                               
      }
    }
*/
   size_t worker_index = try_to_find_worker(in_task);
   std::cerr << "WHOAMIID = " << std::this_thread::get_id() << std::endl;
   std::cerr << "winddex = " << worker_index << std::endl;
//   std::cerr << "worker_index = " << worker_index << "m_workers size = " << m_workers.size() << std::endl;
//   std::cerr << "WHOAMIID = " << std::this_thread::get_id(); 
//   std::cerr << "WHOAMI = " << whoami(); 

   if (in_task.done() && worker_index >= 0) { 
	std::cerr << "condition!!" << std::endl; 
	//m_workers[whoami()].work();
	m_workers[0].work();
   }
   else { 
	std::cerr << "eval only " << std::endl; 
	in_task.run_task(); 
   } 

// && worker_index >= 0) { std::cerr << "done -- now work "; m_workers[worker_index].work(); }

}


bool MultiThreading::ThreadPool::is_pool_full () {
    // returns true if size of pool >= max pool size (usually setted externally)
    return (m_workers.size() >= m_max_thread_number);
}

bool MultiThreading::Worker::is_free () { 
	return task_stack->size() == 0; 
}

void MultiThreading::Worker::add_to_task_stack(Task task) { 
	task_stack->push(task);
	std::cerr << "Size of stack now is " << task_stack->size() << std::endl;
}


