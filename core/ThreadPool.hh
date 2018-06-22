#ifndef GNATHREADPOOL_H
#define GNATHREADPOOL_H

#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include <stack>
#include <condition_variable> 
#include "TransformationBase.hh"
#include "GNAObject.hh"

//using namespace TransformationTypes;


namespace MultiThreading {
  class Worker;

  /**
  * Separate task. Contains pointer to transformation that runs on demand. 
  */
  class Task {
  public:
    Task() {}
    Task(TransformationTypes::Entry *in_entry) : m_entry( in_entry ) { } 

    void operator=(const Task& task) {
	this->m_entry = task.m_entry;
    }

    void run_task();
    inline bool done() { std::cout << "src size " << m_entry->sources.size() << " "; return (!(m_entry->tainted) || (m_entry->sources.size() == 0)); }
//  private:
    TransformationTypes::Entry *m_entry;
  };


  class ThreadPool {
  public:
    ThreadPool (int maxthr = 0);
    ~ThreadPool () {
	std::cout << "thr num = " << threads.size() <<std::endl;
	for (auto &thr : threads) {
	    thr.join();
	}
    }

    int whoami();
    void add_task(Task task);
    void new_worker(Task &task, size_t index);
    int is_free_worker_exists();
    bool is_pool_full();
    void manage_not_motherthread(Task in_task);
    size_t try_to_find_worker(Task in_task);

    size_t worker_count;
//    void set_max_thread_num(int k);

//  private:
    std::vector< Worker > m_workers = {};  // every worker has it's own task stack
    std::vector< std::thread > threads;
    std::vector< std::vector<Task> > m_global_wait_list;
    size_t m_max_thread_number;
    std::mutex tp_add_mutex;
    std::condition_variable cv_tp_add_mutex;
//    std::condition_variable cv_add;
    std::mutex tp_thr_add_mutex;
//    std::condition_variable cv_thr_add;
    std::mutex tp_waitlist_mutex;
//    std::condition_variable cv_waitlist;
    std::mutex tp_pop_mutex;
//    std::condition_variable cv_pop;

  };


  class Worker {
  public:
    Worker(ThreadPool &in_pool);
    void work();
    inline bool is_free () { return task_stack->size() == 0; }
    inline void add_to_task_stack(Task task) { task_stack->push(task); std::cerr << "Size of stack now is " << task_stack->size() << std::endl;}

//private:
    ThreadPool &pool;
    std::thread::id thr_head;
//  std::vector<std::thread::id> mother_thread_ids;
    std::stack<Task> *task_stack;
    
  };
}
#endif /* GNATHREADPOOL_H */
