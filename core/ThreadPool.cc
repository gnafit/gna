#include "ThreadPool.hh"
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <numeric>
#include <thread>
#include <vector>
#include "GNAObject.hh"
#include "TransformationEntry.hh"

// struct Entry;

bool MultiThreading::Task::done() {
	std::cerr << "DONE FUNC src size " << m_entry->sources.size()
		  << std::endl;
	return (m_entry->running || !(m_entry->tainted) || (m_entry->sources.size() == 0));
}

void MultiThreading::Task::run_task() {
	using namespace std::chrono_literals;
	// std::unique_lock<std::mutex> t_lck(task_mtx);
	//if (task_mtx.try_lock()) {
		if (m_entry->tainted && !m_entry->frozen && !m_entry->running) {
			std::cerr << "RUNNING STAT = " << m_entry->running
				  << std::endl;
		//	auto now = std::chrono::system_clock::now();
			std::cerr << "waiting......................"
				  << std::endl;
		//	task_cv.wait_until(t_lck, now + 100ms, [this] {
		//		return !m_entry->running;
		//	});

			if (m_entry->tainted) {  // && !m_entry->frozen) {
				m_entry->mark_running();
				if (m_entry->tainted) {
					m_entry->evaluate();
					std::cerr << "EVA-id = "
						  << std::this_thread::get_id()
						  << std::endl;
					m_entry->tainted = false;
				}
				m_entry->mark_not_running();
			}
		}
		//t_lck.unlock();
	//	task_mtx.unlock();
		//task_cv.notify_one();
	//}
}

MultiThreading::Worker::Worker(ThreadPool &in_pool) : pool(in_pool) {
	task_stack = new std::stack<Task>();
}

void MultiThreading::Worker::work() {  // runs task stack
	Task current_task;
//	std::lock_guard<std::mutex> task_stack_lock(mtx_worker);
	mtx_worker.lock();
	if (task_stack->size() != 0) {
		current_task = task_stack->top();
		task_stack->pop();
	}
	mtx_worker.unlock();
	if (pool.stopped) return;
	current_task.run_task();  // TODO: make it in separate thread
				  //    }
				  //     w_lck.unlock();
}

void MultiThreading::ThreadPool::stop() {
	stopped = true;
	stop_condition.notify_all();
	std::cerr << "thr num = " << threads.size() << std::endl;
	for (size_t i = 0; i < m_workers.size(); i++) {
		std::cerr << i << "thr: " << m_workers[i].thr_head << std::endl;
	}
/*	for (size_t i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
*/	std::cerr << "stopped" << std::endl;
}

MultiThreading::ThreadPool::ThreadPool(int maxthr)
    : m_max_thread_number(maxthr), stopped(false) {
	// Mother thread creation
	std::cerr << "Thread pool created" << std::endl;
	if (m_max_thread_number <= 0)
		m_max_thread_number = std::thread::hardware_concurrency();
//	m_workers.emplace_back(Worker(*this));
//	m_workers[0].thr_head = std::this_thread::get_id();
}

int MultiThreading::ThreadPool::whoami() {
	std::cerr << "in-whoami" << std::endl;
	int tmp = -1;
	// tp_m_workers_mutex.lock();
	std::unique_lock<std::mutex> lock(tp_m_workers_mutex);
	int thr_size = (int)m_workers.size();
	for (int i = 0; i < thr_size; i++) {
		if (m_workers[i].thr_head == std::this_thread::get_id()) {
			lock.unlock();
			tmp = i;
		}
	}
	std::cerr << tmp << " unlck " << std::endl;
	lock.unlock();
	std::cerr << " unlck " << std::endl;
	return tmp;
}

void MultiThreading::ThreadPool::new_worker(MultiThreading::Task &in_task,
					    size_t index) {
	//	std::lock_guard<std::mutex> lock(tp_m_workers_mutex);
	tp_m_workers_mutex.lock();

	m_workers.emplace_back(Worker(*this));
	m_workers[index].thread_index_in_pool = index;
	m_workers[index].thr_head = std::this_thread::get_id();
	std::cerr << "New worker ID = " << std::this_thread::get_id()
		  << std::endl;

	tp_m_workers_mutex.unlock();

// TODO: is it in new thread??
	if (in_task.done())
		m_workers[whoami()].work();
	else
		in_task.run_task();
}

void MultiThreading::ThreadPool::manage_not_motherthread(
    MultiThreading::Task in_task) {
	size_t src_size = in_task.m_entry->sources.size();
	if (src_size > 1) {
		for (size_t i = 1; i < src_size;
		     i++) {  // Try to make new thread
			std::cerr << "SECOND SOURCE! " << i << " size "
				  << src_size << std::endl;
			if (in_task.m_entry->sources[i].sink->entry->tainted) {
				add_task(Task(
				    in_task.m_entry->sources[i].sink->entry));
			}
		}
	}
}

size_t MultiThreading::ThreadPool::try_to_find_worker(
    MultiThreading::Task in_task) {
	int worker_index = -1;

	size_t w_size;
	{
		std::lock_guard<std::mutex> lock(tp_m_workers_mutex);
		w_size = m_workers.size();
	}

	auto curr_id = std::this_thread::get_id();
	size_t iter = 0;
	while (iter < w_size && worker_index == -1) {
		// std::lock_guard<std::mutex> lock(tp_m_workers_mutex);
		tp_m_workers_mutex.lock();
		if ((m_workers[iter].is_free())) {
			{
				m_workers[iter].add_to_task_stack(in_task);
				m_workers[iter].thr_head = curr_id;
				worker_index = iter;
				std::cerr
				    << "Fount worker index = " << worker_index
				    << std::endl;
			}
		}
		iter++;
		tp_m_workers_mutex.unlock();
	}
	std::cerr << "DEBUG TMP " << worker_index << std::endl;

	if (worker_index == -1) {
		if (w_size < m_max_thread_number) {
			std::cerr << "THREADS!! " << std::endl;
			std::lock_guard<std::mutex> lock(tp_threads_mutex);
			std::lock_guard<std::mutex> lk(tp_m_workers_mutex);
			threads.emplace_back(
			    std::thread([&in_task, w_size, this]() {
				    MultiThreading::ThreadPool::new_worker(
					std::ref(in_task), w_size - 1);
			    }));
			std::cerr << "Waiting-end!!!" << std::endl;
			worker_index = threads.size() - 1;
		} else {
			for (size_t i = 0; i < w_size; i++) {
				std::lock_guard<std::mutex> lock(
				    tp_waitlist_mutex);
				if (m_workers[i].thr_head == curr_id &&
				    !in_task.done()) {
					std::cerr
					    << "Task added to wait list, "
					       "m_global_wait_list size = "
					    << m_global_wait_list.size()
					    << " i = " << i << std::endl;
					m_global_wait_list[i].emplace_back(
					    in_task);
				}
			}
		}
	}
	std::cerr << "try-to-find worker_index = " << worker_index
		  << " size =" << threads.size() << " " << std::endl;
	return worker_index;
}

void MultiThreading::ThreadPool::add_task(MultiThreading::Task in_task) {
	size_t src_size = in_task.m_entry->sources.size();
	
	size_t curr_task_worker = add_to_free_worker(in_task);
	if (src_size > 0) {
		Task child_task(std::ref(in_task.m_entry->sources[0].sink->entry));
		add_to_N_worker(child_task, curr_task_worker);
	}

	for (size_t i = 1; i < src_size; i++) {
		if ( in_task.m_entry->sources[i].sink->entry->tainted) {
			Task child_task(std::ref(in_task.m_entry->sources[i].sink->entry));
			size_t n_worker = add_to_free_worker(child_task);
		}			
	}
	
	// TODO barrier	
	//size_t worker_index = try_to_find_worker(in_task);
	in_task.run_task(); // TODO to the find worker??? 
		

/*	if (in_task.done() && worker_index >= 0) {
		std::cerr << "condition!!" << std::endl;
		m_workers[whoami()].work();
		//		m_workers[0].work();
	} else {
		std::cerr << "eval only " << std::endl;
		in_task.run_task();
	}
*/
}

bool MultiThreading::ThreadPool::is_pool_full() {
	// returns true if size of pool >= max pool size (usually setted
	// externally)
	return (m_workers.size() >= m_max_thread_number);
}

bool MultiThreading::Worker::is_free() { return task_stack->size() == 0; }

void MultiThreading::Worker::add_to_task_stack(Task task) {
	task_stack->push(task);
	std::cerr << "Size of stack now is " << task_stack->size() << std::endl;
}


void MultiThreading::Worker::wait() { 
	
}

