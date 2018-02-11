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

#include "TransformationBase.hh"
//#include "Transformation.hh"
#include "GNAObject.hh"

//#include "Parameters.hh"
//#include "Data.hh"

//#include "Transformation.hh"
//#include "Exceptions.hh"

//using namespace TransformationTypes;


namespace MultiThreading {
class Worker;

class Task {
public:
    Task(TransformationTypes::Entry *in_entry) { entry = in_entry; }
    void run_task();
    TransformationTypes::Entry *entry;
    /*  TransformationTypes::Args args;
      TransformationTypes::Rets rets;
      TransformationTypes::Function func;
    */
};


class ThreadPool {
public:
  ThreadPool (int maxthr = 0);
  void add_task(Task task);
  int is_free_worker_exists();
  bool is_pool_full();
  void set_max_thread_num(int k);

//private:
  std::vector< Worker > workers = {};
  std::vector< std::thread > pool = {};
  std::vector<int> pool_task_counters ={};
  size_t max_thread_number = 0;
};


class Worker {
public:
  Worker(ThreadPool &in_pool);
  void work();
  void wait_for_free_thread();
  bool is_any_fakes_in_pool();
  bool is_free ();
  bool is_fake();

//private:
  ThreadPool &pool;
 // std::thread thread;
//  std::vector<std::thread::id> mother_thread_ids;
  std::stack<Task> task_stack;
  bool freedom = true;
  bool isfake = false;
};

/*class ThreadPool {
public:
  ThreadPool (int maxthr = 0);
  void add_task(Function in_func);
  int is_free_worker_exists();
  bool is_pool_full();
  void set_max_thread_num(int k);

private:
  std::vector< Worker* > workers = {};
  std::vector< std::thread > pool = {};
  std::vector<int> pool_task_counters ={};
  int max_thread_number;
};
*/
}
#endif /* GNATHREADPOOL_H */
