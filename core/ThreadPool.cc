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


/*
 * Run a single task by evaluation of corresponding transformation.
 * If transformation is running or finished already, nothing is done. 
 *
 */
void MultiThreading::Task::run_task() {
	// TODO check is task already ready ro run (all inputs are valid)
	// if not -- wait until ready
	if (m_entry->running || !m_entry->tainted) return;	

	m_entry->mark_running();

	m_entry->evaluate();
	m_entry->tainted = false;

	m_entry->mark_not_running();
}

MultiThreading::Worker::Worker(ThreadPool &in_pool) : pool(in_pool) {
	task_stack =  std::stack<Task>();
}


/*
 * Wake up worker, change its status to WorkerStatus::Run, and run tasks from worker's task stack.
 * When stack becomes empty, worker changes its status to WorkerStatus::Sleep.
 *
 */
void MultiThreading::Worker::work() {  // runs task stack
	worker.status = WorkerStatus::Run;
	worker.wakeup();

	size_t w_size = worker.get_stack_size();
	while (!task_stack.empty()) {
		task_stack.top().run_task();
		task_stack.pop();
	}

	worler.sleep();
	worker.status = WorkerStatus::Sleep;
}

void MultiThreading::ThreadPool::stop() {
	stopped = true;
	//stop_condition.notify_all();
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
//					m_global_wait_list[i].emplace_back(
//					    in_task);
				}
			}
		}
	}
	std::cerr << "try-to-find worker_index = " << worker_index
		  << " size =" << threads.size() << " " << std::endl;
	return worker_index;
}


/*
 * Add task to the task stack of the Worker number of N.
 * Check correctness of input N. If there is no N-th thread, throw exception.
 * Lock an access to MultiThreading::ThreadPool::m_workers
 *
 */
void MultiThreading::ThreadPool::add_to_N_worker(MultiThreading::Task in_task, size_t N) {
	tp_m_workers_mutex.lock();
	if (get_workers_count() <= N) {
		throw std::runtime_error("Not enough workers.");
	}
	m_workers[N].task_stack.push(in_task);
	tp_m_workers_mutex.unlock();
}


/*
 * Find the worker with an empty stack.
 * If such such worker is found, push the task into its stack.
 * If there is no free Workers, push the task into global wait list.
 */
int MultiThreading::ThreadPool::add_to_free_worker(MultiThreading::Task in_task) {
	
	int i = 0; 

	tp_m_workers_mutex.lock();
	size_t n_workers = get_workers_count();
	while (i < static_cast<int>(n_workers) && m_workers[i].status != WorkerStatus::Sleep) {
		 i++;
	}	
	tp_m_workers_mutex.unlock();

	if (i == static_cast<int>(n_workers)) {
		add_to_global_wait_list(in_task);
		return -1; // if free worker was not found
	}
	
	add_to_N_worker(in_task, i);
	return i;
}

/*
 * Push task into global wait list (single per thread pool)
 * 
 *
 */
void MultiThreading::ThreadPool::add_to_global_wait_list(MultiThreading::Task in_task) {
	tp_waitlist_mutex.lock();
	m_global_wait_list.push(in_task);
	tp_waitlist_mutex.unlock();
}

size_t MultiThreading::ThreadPool::get_workers_count() {
	// TODO add locks
	return m_workers.size();
}


/*
 * Run task (transformation) at free thread and evaluate its children.
 * 
 *
 */
void MultiThreading::ThreadPool::add_task(MultiThreading::Task in_task, int entry_point_stat) {
	size_t src_size = in_task.m_entry->sources.size();
	
	size_t iter = 1;
	bool ready_to_run = true;

	if (entry_point_stat < 0) {
		iter = 0;
		int curr_task_worker = add_to_free_worker(in_task); // if -1, it is already added to the wait list
		if (curr_task_worker != -1 && src_size > 0 && !(in_task.m_entry->sources[0].sink->entry->running) && in_task.m_entry->sources[0].sink->entry->tainted)  {
			Task child_task(std::ref(in_task.m_entry->sources[0].sink->entry));
			add_to_N_worker(child_task, curr_task_worker);
			add_task(child_task, curr_task_worker);
			ready_to_run = false;
		}
	}
 
	for ( ; iter < src_size; iter++) {
		if ( in_task.m_entry->sources[iter].sink->entry->tainted && !(in_task.m_entry->sources[iter].sink->entry->running)) {
			Task child_task(std::ref(in_task.m_entry->sources[iter].sink->entry));
			int n_worker = add_to_free_worker(child_task);
			if (n_worker != -1) add_task(child_task, n_worker);
			ready_to_run = false;
		}			
	}

	if (ready_to_run) {
		// TODO how to write runtask for w_is = -1
		in_task.run_task();
	}

		
	// TODO barrier	
	//size_t worker_index = try_to_find_worker(in_task);

}

bool MultiThreading::ThreadPool::is_pool_full() {
	// returns true if size of pool >= max pool size (usually setted
	// externally)
	return (m_workers.size() >= m_max_thread_number);
}

bool MultiThreading::Worker::is_free() { return task_stack.size() == 0; }

void MultiThreading::Worker::add_to_task_stack(Task task) {
	task_stack.push(task);
	std::cerr << "Size of stack now is " << task_stack.size() << std::endl;
}

/*
 * Take one task from global queue
 *
 */
void MultiThreading::Worker::bite_global_wait_list() {
	pool.tp_waitlist_mutex.lock();
	// Yeah, there is an extra checking if it is empty.
	// But it has to be here as someone can bite wait list before you
	// from other thread
	if (! pool.m_global_wait_list.empty()) {
		Task curr_task = pool.m_global_wait_list.front();
		pool.m_global_wait_list.pop();
		curr_task.run_task();
	}
	pool.tp_waitlist_mutex.unlock();
	sleep();
}

/*
 * Keep thread sleeping until there is no task for it.
 * Stop sleeping in cases of:
 * - local task stack becomes not empty
 * - global wait list becomes not empty
 * 
 */
void MultiThreading::Worker::sleep() { 
	std::unique_lock<std::mutex> 
                lock(cv_worker);
             
	if( !pool.stopped && pool.m_global_wait_list.empty() && task_stack.empty() ) {
        	cv_worker.wait(lock);
		wakeup();
	} else if (! pool.m_global_wait_list.empty() ) {
		bite_global_wait_list();
	}
	//wakeup();
}
