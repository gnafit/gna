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
    this->entry->evaluate();
}


MultiThreading::Worker::Worker(ThreadPool &in_pool) : pool(in_pool){
//    if (pool.pool.size() < pool.workers.size()) isfake = true;    
/*    if (!isfake) {
      mother_thread_ids.push_back(std::this_thread::get_id());
    }*/
}

void MultiThreading::Worker::work(){ // runs task stack
    while (pool.size() > 0) {
// TODO: add lock task stack
      Task current_task = task_stack.top();
      task_stack.pop(); 
//TODO: add unlock task stack
      current_task.run_task(); // TODO: make it in separate thread
    }
}

void MultiThreading::Worker::wait_for_free_thread() {
}

bool MultiThreading::Worker::is_any_fakes_in_pool() {
    return (pool.pool.size() < pool.workers.size());
}

bool MultiThreading::Worker::is_free () {
    return freedom;
}

bool MultiThreading::Worker::is_fake() {
    return isfake;
}

MultiThreading::ThreadPool::ThreadPool (int maxthr) : max_thread_number(maxthr) {
    // Mother thread creation
    // setting max thread number
    std::cout << "Mother thread created" << std::endl;
    if (max_thread_number <= 0) max_thread_number = std::thread::hardware_concurrency();
    workers.push_back(Worker(*this));
    //pool.push_back(std::thread(workers[0]));
   // pool_task_counters.push_back(0);
}

void MultiThreading::ThreadPool::add_task(MultiThreading::Task in_task) {
    // mother thread  creates new one
    if (!is_pool_full()) {
      if (! is_free_worker_exists() ) {
        workers.push_back(Worker(*this));
        //pool_task_counters.push_back(0);
      }
    }
}

int MultiThreading::ThreadPool::is_free_worker_exists() {
    return 0;
    // returns index of free worker
    // if there is no free workers, returns -1
} 

bool MultiThreading::ThreadPool::is_pool_full () {
    // returns true if size of pool >= max pool size (usually settet externally)
    return (pool.size() < max_thread_number);
}

void MultiThreading::ThreadPool::set_max_thread_num(int k) {
    max_thread_number = k;
}
