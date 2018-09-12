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
#include <queue>
#include <condition_variable> 
#include "TransformationEntry.hh"
#include "GNAObject.hh"

//using namespace TransformationTypes;


namespace MultiThreading {

 /*
  * Show status of the thread corresponds to MultiThreading::Worker 
  * 
  *
  */
  enum class WorkerStatus { 
    Sleep = 0,		///< Free worker, may be woke up and used
    InTheWings, 	///< Worker is still sleeping but stack is not empty -- expected to run soon  
    Run,		///< Worker has tasks in its task stack
    Stopped,		///< Worker already stopped, can't be used
    Crashed		///< Smth wrong 
  };

  class Worker;

  /**
  * Separate task. Contains pointer to transformation that runs on demand. 
  */
  class Task {
  public:
    Task() {}
    Task(TransformationTypes::Entry *in_entry) : m_entry( in_entry ) { } 
    Task(Task &&task) : m_entry(std::move(task.m_entry)) {}
    Task(const Task &task) : m_entry(task.m_entry) {}

    void operator=(const Task& task) {
	this->m_entry = task.m_entry;
    }

    void run_task();
//  private:
    TransformationTypes::Entry *m_entry;

    size_t worker_index = -1;

// TODO DELETE
    std::mutex task_mtx;
    std::condition_variable task_cv;
  };


  class ThreadPool {
  public:
    ThreadPool (int maxthr = 0);
    ~ThreadPool () {
	stop();
    }

    void stop();

    int whoami();
    void add_task(Task& task, int entry_point_stat = -1);
    void new_worker(Task &task);
    bool is_pool_full();
    size_t try_to_find_worker(Task& in_task);

    void add_to_N_worker(MultiThreading::Task& in_task, size_t N);
    int add_to_free_worker(MultiThreading::Task& in_task);
    void add_to_global_wait_list(MultiThreading::Task& in_task);
    size_t get_workers_count();


    size_t worker_count;
    size_t m_max_thread_number;

    bool stopped;

//  private:
    std::vector< Worker > m_workers = {};  // every worker has it's own task stack
    std::vector< std::thread > threads;
    std::queue< Task > m_global_wait_list;


    std::mutex tp_m_workers_mutex;
    std::mutex tp_threads_mutex;
    std::mutex tp_waitlist_mutex;

//    std::condition_variable cv_tp_m_workers_mutex;
//    std::condition_variable stop_condition;
    std::condition_variable cv_global_wait_list;
  };


  class Worker {
  public:
    Worker(ThreadPool &in_pool);
    Worker(Worker &&worker) : pool(std::move(worker).pool), thr_head(std::move(worker.thr_head)), task_stack(std::move(worker.task_stack)) { }
    void work();
    bool is_free ();
    void add_to_task_stack(Task& task);
    void sleep();
    void wakeup();
    size_t get_stack_size();

//private:
    ThreadPool &pool;
    std::thread::id thr_head;
    size_t thread_index_in_pool = 0;
    WorkerStatus status = WorkerStatus::Sleep;
//    std::thread w_thread;
//  std::vector<std::thread::id> mother_thread_ids;
    std::stack<Task> task_stack ;

    mutable std::mutex mtx_worker;
    std::condition_variable cv_worker;    
  };
}
#endif /* GNATHREADPOOL_H */
