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

//using namespace TransformationTypes;

void MultiThreading::Task::run_task() {
    printf("runtask\n");
    if (m_entry->tainted) {
	m_entry->evaluate();
    }
    m_entry->tainted = false;
}


MultiThreading::Worker::Worker(ThreadPool &in_pool) : pool(in_pool) {
   // thr_head = std::this_thread::get_id();
    task_stack = new std::stack<Task>();
}


void MultiThreading::Worker::work(){ // runs task stack
    std::cerr << "Work" << std::endl;
    while (!task_stack->empty()) {

    std::cout << "Work task size  = " << task_stack->size() << std::endl;
// TODO: add lock task stack
      Task& current_task = task_stack->top();
//      if (!current_task.ready()) { /* make waiting for finish children */ }
      task_stack->pop(); 
//TODO: add unlock task stack
      current_task.run_task(); // TODO: make it in separate thread
    }
}


MultiThreading::ThreadPool::ThreadPool (int maxthr) : m_max_thread_number(maxthr) {
    // Mother thread creation
    std::cout << "Thread pool created" << std::endl;
    if (m_max_thread_number <= 0) m_max_thread_number = std::thread::hardware_concurrency();
    m_workers.push_back(Worker(*this));
    m_workers[0].thr_head = std::this_thread::get_id();
}


void MultiThreading::ThreadPool::new_worker(MultiThreading::Task &in_task, size_t index) {
//        std::lock_guard<std::mutex> lock(tp_waitlist_mutex);
	m_workers.push_back(Worker(*this));
	m_global_wait_list.push_back({});
        std::cerr << "New worker added!" << std::endl;
        m_workers[index].thr_head =  std::this_thread::get_id();
	std::cout << "New worker ID = " << std::this_thread::get_id() << std::endl;
        if (in_task.done()) m_workers[index].work();
}

void MultiThreading::ThreadPool::add_task(MultiThreading::Task in_task) {
    // mother thread  creates new one if it is possible
    std::thread::id curr_id = std::this_thread::get_id();
    bool worker_found = false;
    size_t w_size; 
    {
        std::lock_guard<std::mutex> lock(tp_waitlist_mutex);
        worker_count = m_workers.size();
        w_size = worker_count;
    }
    size_t worker_index = 0;

    for (size_t i = 0; i < w_size; i++)  {
      std::lock_guard<std::mutex> lock(tp_add_mutex);
      if((m_workers[i].is_free())) { // || (!m_workers[i].is_free() && curr_id == m_workers[i].thr_head)) { 
        std::cerr << "Free worker found!" << std::endl;
	{
        	m_workers[i].add_to_task_stack(in_task);
        	m_workers[i].thr_head = curr_id;
        	worker_found = true;
		worker_index = i;
	}
        break;
      }
    }

    if (!worker_found) {
      std::lock_guard<std::mutex> lock(tp_add_mutex);
      if (w_size < m_max_thread_number) {
	threads.push_back(std::thread([&in_task, w_size, this]() {MultiThreading::ThreadPool::new_worker (std::ref(in_task), w_size - 1); }) );
      } else {
        std::cerr << "Try to add task to wait list, w_size =  " << w_size 
			<< " m_workers = " << m_workers.size()  << std::endl;
        for (size_t i = 0; i < w_size; i++) {
	  std::cout << i << " " << m_workers[i].thr_head << " ";
          if (m_workers[i].thr_head == curr_id && !in_task.done()) {
            std::cerr << "Task added to wait list, m_global_wait_list size = " 
			<< m_global_wait_list.size() << " i = " << i  << std::endl;
            m_global_wait_list[i].push_back(in_task);
          }
        }
      } 
    }
   
    std::cout << "w ind = " << worker_index << std::endl;
    size_t src_size = in_task.m_entry->sources.size();                                               
// TODO move to top -- always to current
    std::cout << "src_size = " << src_size << std::endl;
    if (in_task.m_entry->sources.size() > 0) {
      if ( in_task.m_entry->sources[0].sink->entry->tainted) {
        Task motherthread_task(in_task.m_entry->sources[0].sink->entry);                                 
    
        add_task(motherthread_task);                                                               
        in_task.m_entry->sources[0].sink->entry->touch();       // first one always runs in the same thread (current main thread for exact entry)
      }
      for (size_t i = 1; i < src_size; i++) {                 // Try to make new thread                
            std::cout << "SECOND SOURCE! " << i << " size " << src_size << std::endl;
            if ( in_task.m_entry->sources[i].sink->entry->tainted) {
                add_task(Task(in_task.m_entry->sources[i].sink->entry));                                       
            }
      }
   }
   if (in_task.done()) { std::cout << "done -- now work "; m_workers[worker_index].work(); }
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

