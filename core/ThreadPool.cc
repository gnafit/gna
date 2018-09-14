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




inline std::ostream& operator<<(std::ostream& so, MultiThreading::WorkerStatus stat) {
	switch(stat) {
		case MultiThreading::WorkerStatus::Sleep: so << "Sleep"; break;	
		case MultiThreading::WorkerStatus::InTheWings: so << "InTheWings"; break;	
		case MultiThreading::WorkerStatus::Run: so << "Run"; break;	
		case MultiThreading::WorkerStatus::Stop: so << "Stop"; break;	
		case MultiThreading::WorkerStatus::Crashed: so << "Crashed"; break;	
	}
	return so;
}



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

	m_entry->tainted = false;
	m_entry->evaluate();

	m_entry->mark_not_running();
}

MultiThreading::Worker::Worker(ThreadPool &in_pool) : pool(in_pool) {
	task_stack =  std::stack<Task>();
	thr_head = std::this_thread::get_id();
	std::cerr << "New worker ID = " << std::this_thread::get_id() << std::endl;
//	sleep();
}


/*
 * Wake up worker, change its status to WorkerStatus::Run, and run tasks from worker's task stack.
 * When stack becomes empty, worker changes its status to WorkerStatus::Sleep.
 *
 */
void MultiThreading::Worker::work() {  // runs task stack
//	status = WorkerStatus::Run;
	wakeup();

	while (!task_stack.empty()) {
		task_stack.top().run_task();
		task_stack.pop();
	}

	sleep();
	//worker.status = WorkerStatus::Sleep;
}

void MultiThreading::ThreadPool::stop() {
	stopped = true;
	//stop_condition.notify_all();
	std::cerr << "thr num = " << m_workers.size() << std::endl;
	for (size_t i = 0; i < m_workers.size(); i++) {
		std::cerr << i << "thr: " << m_workers[i].thr_head << " " << m_workers[i].status << std::endl;
	}
/*	for (size_t i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
*/	std::cerr << "stopped" << std::endl;
}

MultiThreading::ThreadPool::ThreadPool(int maxthr)
    : m_max_thread_number(maxthr), stopped(false) {
	std::cerr << "Thread pool created" << std::endl;
	if (m_max_thread_number <= 0)
		m_max_thread_number = std::thread::hardware_concurrency();
	m_workers.emplace_back(Worker(*this));
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


void MultiThreading::ThreadPool::new_worker(MultiThreading::Task& in_task) {
	//	std::lock_guard<std::mutex> lock(tp_m_workers_mutex);
	tp_m_workers_mutex.lock();
	std::cout << "Add new worker!" << std::endl;
	m_workers.emplace_back(Worker(*this));

	tp_m_workers_mutex.unlock();
}


/*
 * Add task to the task stack of the Worker number of N.
 * Check correctness of input N. If there is no N-th thread, throw exception.
 * Lock an access to MultiThreading::ThreadPool::m_workers
 *
 */
void MultiThreading::ThreadPool::add_to_N_worker(MultiThreading::Task& in_task, size_t N) {
	tp_m_workers_mutex.lock();
	if (get_workers_count() < N) {
		std::cerr << "workers count = " << get_workers_count() << ", N = " << N <<std::endl;
		throw std::runtime_error("GNA Thread Pool ERROR: Not enough workers.");
	}
	m_workers[N].task_stack.push(in_task);
	m_workers[N].status = WorkerStatus::InTheWings;
	tp_m_workers_mutex.unlock();
}


/*
 * Find the worker with an empty stack.
 * If such such worker is found, push the task into its stack.
 * If there is no free Workers, push the task into global wait list.
 */
int MultiThreading::ThreadPool::add_to_free_worker(MultiThreading::Task& in_task) {
	
	int i = 0; 

	tp_m_workers_mutex.lock();
	size_t n_workers = get_workers_count();
	while (i < static_cast<int>(n_workers) && m_workers[i].status != WorkerStatus::Sleep) {
		 i++;
	}	
	std::cout << "i = " << i << ", n_workers = " << n_workers << std::endl;
	std::cout << "m_max_thread_number = " << m_max_thread_number << std::endl;
	if (i == static_cast<int>(n_workers-1)) {
		if (m_max_thread_number > n_workers) {
			tp_threads_mutex.lock();	
			threads.emplace_back(std::thread([&in_task, this]() {
				    MultiThreading::ThreadPool::new_worker(in_task);
			    }));
			tp_threads_mutex.unlock();
		} else {
			add_to_global_wait_list(in_task);
			return -1; // if free worker was not found
		}
	}
	tp_m_workers_mutex.unlock();
	
	add_to_N_worker(in_task, i);
	return i;
}

/*
 * Push task into global wait list (single per thread pool)
 * 
 *
 */
void MultiThreading::ThreadPool::add_to_global_wait_list(MultiThreading::Task& in_task) {
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
void MultiThreading::ThreadPool::add_task(MultiThreading::Task& in_task, int entry_point_stat) {
	size_t src_size = in_task.m_entry->sources.size();
	std::cerr << "Add_task worker ID = " << std::this_thread::get_id() << std::endl;	
	std::cerr << "Entry_point_stat = " << entry_point_stat << std::endl;
	size_t iter = 1;
	bool ready_to_run = true;

	if (entry_point_stat < 0) {
		iter = 0;
		int curr_task_worker = add_to_free_worker(in_task); 
		// if -1, it is already added to the wait list
		auto& child_entry = in_task.m_entry->sources[0].sink->entry;
		if (curr_task_worker != -1 && src_size > 0 && 
				!(child_entry->running) && child_entry->tainted)  {
			Task child_task(std::ref(in_task.m_entry->sources[0].sink->entry));
			add_to_N_worker(child_task, curr_task_worker);
			add_task(child_task, curr_task_worker);
			ready_to_run = false;
		}
	}
 
	for ( ; iter < src_size; iter++) {
		auto& child_entry = in_task.m_entry->sources[iter].sink->entry;
		if ( child_entry->tainted && !(child_entry->running)) {
			Task child_task(std::ref(child_entry));
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

void MultiThreading::Worker::add_to_task_stack(Task&  task) {
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
	std::cerr << "Sleeping....................................................." << std::endl; 
	status = WorkerStatus::Sleep;
	std::unique_lock<std::mutex> 
                lock(mtx_worker);
             
	if( !pool.stopped && pool.m_global_wait_list.empty() && task_stack.empty() ) {
        	cv_worker.wait(lock);
		wakeup();
	} else if (! pool.m_global_wait_list.empty() ) {
		bite_global_wait_list();
	}
	//wakeup();
}

void MultiThreading::Worker::wakeup() {
	std::cerr << "WAKING UP!!!" << std::endl;
        status = WorkerStatus::Run;
	// TODO notification?
}
