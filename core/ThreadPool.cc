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
#include <mutex>
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


void MultiThreading::Task::count_all_inputs() {
	inputs = m_entry->sources.size();
	ready_inputs = 0;
	for (size_t i = 0 ; i < inputs; i++) {
		if (! m_entry->sources[0].sink->entry->tainted)
			ready_inputs++;
	}	
	std::cout << "inputs = " << inputs << ", ready_inputs = " << ready_inputs << std::endl;
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

	std::cerr << "TASK RUN!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

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
//	wakeup();
	
	std::cerr << "WWWWWWOOOOORRRRRRKKKK worker ID = " << std::this_thread::get_id() << std::endl;	
	//while (!task_stack.empty()) {
	while (task_stack.size() > 0) {
	
		std::cerr << "BEFORE BEFORE POP, stack size = " << task_stack.size() << std::endl;
		task_stack.top().run_task();
		std::cerr << "BEFORE POP, stack size = " << task_stack.size() << std::endl;
		if (task_stack.size() > 0)  task_stack.pop(); // we need it in case of new chain from runtask
		pool.active_tasks--;
	}

//	sleep(); // TODO Uncomment
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
*/
	std::cerr << "stopped" << std::endl;
}

MultiThreading::ThreadPool::ThreadPool(int maxthr)
    : m_max_thread_number(maxthr), stopped(false), active_tasks(0) {
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
			//lock.unlock();
			tmp = i;
		}
	}
	std::cerr << tmp << " unlck " << std::endl;
	lock.unlock();
	std::cerr << " unlck " << std::endl;
	return tmp;
}


void MultiThreading::ThreadPool::new_worker(MultiThreading::Task& in_task) {
	std::lock_guard<std::mutex> lock(tp_m_workers_mutex);
//	tp_m_workers_mutex.lock();
	std::cout << "Add new worker!" << std::endl;
	m_workers.emplace_back(Worker(*this));

//	tp_m_workers_mutex.unlock();
}


/*
 * Add task to the task stack of the Worker number of N.
 * Check correctness of input N. If there is no N-th thread, throw exception.
 * Lock an access to MultiThreading::ThreadPool::m_workers
 *
 */
void MultiThreading::ThreadPool::add_to_N_worker(MultiThreading::Task& in_task, size_t N) {
	std::lock_guard<std::mutex> lock1(tp_m_workers_mutex);
	//std::lock(lock1);
	if (get_workers_count() < N) {
		throw std::runtime_error("GNA Thread Pool ERROR: Not enough workers.");
	}
	std::lock_guard<std::mutex> lock2(m_workers[N].mtx_worker);
	m_workers[N].task_stack.push(in_task);
	m_workers[N].status = WorkerStatus::InTheWings;
//	tp_m_workers_mutex.unlock();
}


/*
 * Find the worker with an empty stack.
 * If such such worker is found, push the task into its stack.
 * If there is no free Workers, push the task into global wait list.
 */
int MultiThreading::ThreadPool::add_to_free_worker(MultiThreading::Task& in_task) {
	
	int i = 0;
	std::cout << "INSIDE ADD_TO_FREE_WORKER" << std::endl; 

	std::unique_lock<std::mutex> lock(tp_m_workers_mutex);
//	std::cout << "INSIDE 2" << std::endl; 
//	std::lock_guard<std::mutex> lock(tp_m_workers_mutex);
	size_t n_workers = get_workers_count();
	while (i < static_cast<int>(n_workers) && m_workers[i].status != WorkerStatus::Sleep) {
		 i++;
	}
	lock.unlock();	
//	std::cout << "INSIDE 3" << std::endl; 
//	std::cout << "i = " << i << ", n_workers = " << n_workers << std::endl;
	if (i == static_cast<int>(n_workers)) {
		if (m_max_thread_number > n_workers) {
//	std::cout << "INSIDE 4" << std::endl; 
			std::unique_lock<std::mutex> tplock(tp_threads_mutex);	
//			std::cout << "INSIDE 5" << std::endl; 
		/*	threads.emplace_back(std::thread([&in_task, this]() {
				    MultiThreading::ThreadPool::new_worker(in_task);
			    }));
		*/
			new_worker(in_task);
			std::cout << "INSIDE 6" << std::endl; 
			tplock.unlock();
		} else {
			std::cout << "INSIDE 7" << std::endl; 
			add_to_global_wait_list(in_task);
			return -1; // if free worker was not found
		}
	}
//	lock.unlock();
	
//			std::cout << "INSIDE 8" << std::endl; 
	add_to_N_worker(in_task, i);
			std::cout << "INSIDE 9" << std::endl; 
	return i;
}

/*
 * Push task into global wait list (single per thread pool)
 * 
 *
 */
void MultiThreading::ThreadPool::add_to_global_wait_list(MultiThreading::Task& in_task) {
	std::unique_lock<std::mutex> lock(tp_waitlist_mutex);
	m_global_wait_list.push(in_task);
	lock.unlock();
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
	active_tasks++;
	if (! entry_point_entry) entry_point_entry = in_task.m_entry;
// TODO add exception
	size_t src_size = in_task.m_entry->sources.size();
//	std::cerr << "Entry_point_stat = " << entry_point_stat << " src_size = " << src_size << std::endl;

//	size_t iter = 1;
	bool children_are_ready = true;
	int curr_task_worker = entry_point_stat;

/*	for (size_t i = 0; i < m_workers.size(); i++) {
                std::cout << "worker " << i << " size: " << m_workers[i].task_stack.size() << std::endl;
        }
*/
	std::cout << "!1111!!!1111111111111111111111entry_point_stat  = " << entry_point_stat << std::endl;
	if (entry_point_stat < 0) {
			// WHY THIS CONDITION!!!!?????
			// to avoid double adding myself
	//	iter = 0;
		curr_task_worker = add_to_free_worker(in_task);
		std::cout << "CURRENT_WORKER = " <<  curr_task_worker << std::endl;  
	}
		// if -1, it is already added to the wait list
		
	auto& child_entry = in_task.m_entry->sources[0].sink->entry;
	
	if (curr_task_worker != -1 && src_size > 0  && 
				!(child_entry->running) && child_entry->tainted)  {
		
	//if ( src_size > 0 )  {
			Task child_task(std::ref(in_task.m_entry->sources[0].sink->entry));
			add_to_N_worker(child_task, curr_task_worker);
			add_task(child_task, curr_task_worker);
			children_are_ready = false;

			std::cout << "child's children number = " << child_task.inputs << std::endl; 
		}

//	iter = 0;
//	std::cout << "before FOR iter = " << iter << " srcsize = " << src_size << std::endl;
/*	for (; iter < src_size; iter++) {
		std::cout << iter << std::endl;
		auto& child_entry = in_task.m_entry->sources[iter].sink->entry;
		if ( child_entry->tainted && !(child_entry->running)) {
			children_are_ready = false;
		}
		Task child_task(std::ref(child_entry));
		std::cout << "!!! 3" << std::endl;
		curr_task_worker = add_to_free_worker(child_task); // SMTH WITH LOCK HERE
		std::cout << "curr_task_worker " << curr_task_worker << std::endl;  

		std::cout << "child's children number = " << child_task.inputs << std::endl; 

		if (curr_task_worker >= 0)  add_task(child_task, curr_task_worker);
		std::cout << " CHILDs CURRENT TASK WORKER = " << curr_task_worker << std::endl;
		//}			
	}
*/	
	for (size_t i = 0; i < m_workers.size(); i++) {
		std::cout << "worker " << i << " size: " << m_workers[i].task_stack.size() << std::endl;  
	}
	std::cout << "READY??? " << children_are_ready << std::endl;	

	std::cout << "waitlist_size = " << m_global_wait_list.size() << std::endl;

	if (children_are_ready) {
		// TODO how to write runtask for w_is = -1
//		std::cerr << "entry_point_stat = " << entry_point_stat << std::endl;
//		std::cerr << "curr_task_worker = " << curr_task_worker << std::endl;
	//	ready_to_run = true;
		
	//	if (entry_point_stat >= 0 ) {
		if (curr_task_worker >= 0 ) {
		//	m_workers[entry_point_stat].ready_to_run = true;

			m_workers[curr_task_worker].ready_to_run = true;

		//	m_workers[entry_point_stat].work();
			m_workers[curr_task_worker].work();
		} 
		/*else if (entry_point_stat >= 0){
			m_workers[entry_point_stat].ready_to_run = true;
			m_workers[entry_point_stat].work();
		}*/
		//else 
		
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

/*MultiThreading::Worker& MultiThreading::Worker::operator=(const MultiThreading::Worker& other) {
	if (this != &other) {
		pool = std::move(other.pool);
		status = other.status;
		task_stack = other.task_stack;
		thr_head = std::move(worker.thr_head);
		task_stack = std::move(worker.task_stack);
	}	

	return *this;
} */


/*
 * Take one task from global queue
 *
 */
void MultiThreading::Worker::bite_global_wait_list() {
	std::lock_guard<std::mutex> lock(pool.tp_waitlist_mutex);
	// Yeah, there is an extra checking if it is empty.
	// But it has to be here as someone can bite wait list before you
	// from other thread
	if (! pool.m_global_wait_list.empty()) {
		Task curr_task = pool.m_global_wait_list.front();
		pool.m_global_wait_list.pop();
		curr_task.run_task();
		pool.active_tasks--;
                std::cout << "active task -- = " << pool.active_tasks << std::endl;

	}
	//pool.tp_waitlist_mutex.unlock();
	//sleep(); // TODO uncomment
}


bool MultiThreading::Worker::is_ready() {
	return ready_to_run;
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
	std::unique_lock<std::mutex> lock(mtx_worker);
             
	if (!pool.entry_point_entry->tainted) pool.stop();
	//if( !pool.stopped) wait(cv_worker, is_ready);

	
/*	if( !pool.stopped && 	pool.m_global_wait_list.empty() && task_stack.empty() ) {
		std::cout << "sleeping before lock" << std::endl;
        	cv_worker.wait(lock);
		std::cout << "sleeping after lock" << std::endl;
//		pool.stop_condition.wait(lock)
		wakeup();
	} else if (! pool.m_global_wait_list.empty() ) {
		bite_global_wait_list();
	}
*/
	
	//wakeup();
}

void MultiThreading::Worker::wakeup() {
	std::cerr << "WAKING UP!!!" << std::endl;
	cv_worker.notify_all();	
        status = WorkerStatus::Run;
	// TODO notification?
}
