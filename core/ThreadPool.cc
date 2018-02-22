#include <chrono>
//#include <mutex>
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
    m_entry->evaluate();
    m_entry->tainted = false;
}


MultiThreading::Worker::Worker(ThreadPool &in_pool) : pool(in_pool) { }


void MultiThreading::Worker::work(){ // runs task stack
    std::cerr << "Work" << std::endl;
    while (!task_stack.empty()) {
// TODO: add lock task stack
      Task current_task = task_stack.top();
//      if (!current_task.ready()) { /* make waiting for finish children */ }
      task_stack.pop(); 
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

void MultiThreading::ThreadPool::add_task(MultiThreading::Task in_task) {
    // mother thread  creates new one if it is possible

    std::thread::id curr_id = std::this_thread::get_id();
    bool worker_found = false;
    for (auto worker : m_workers) {
      if(worker.is_free()) { 
        std::cerr << "Free worker found!" << std::endl;
        worker.add_to_task_stack(in_task);
        worker.thr_head = curr_id;
        worker_found = true;
        //if (in_task.done()) 
        worker.work();
        break;
      }
    }
    if (!worker_found) {
      size_t w_size = m_workers.size();
      if (w_size < m_max_thread_number) {
        m_workers.push_back(Worker(*this));
        std::cerr << "New worker added!" << std::endl;
        m_workers[w_size - 1].thr_head = curr_id;
        if (in_task.done()) m_workers[w_size - 1].work();
      } else {
        std::cerr << "Try to add task to wait list" << std::endl;
        for (size_t i = 0; i < w_size; i++) {
          if (m_workers[i].thr_head == curr_id && !in_task.done()) {
            std::cerr << "Task added to wait list" << std::endl;
            m_global_wait_list[i].push_back(in_task);
          }
        }
      } 
    }
    
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
