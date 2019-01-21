#include "Scheduler.hh"


void LocalList::fill(TransformationTypes::Entry* entry) {
	key_task = Task(entry, this);
	size_t src_size = entry->sources.size();
	for (size_t i = 0; i < src_size; i++) {
		panding.emplace_back(Task(entry->ources[i].sink->entry, this));
	}
	// TODO add global list def
}

void GlobalList::add(LocalList list) {
	local_lists.emplace_back(list);
}

void Task::execute_safe() { 
	// TODO add locks
	execute();
	// TODO add unlock
}

Task* LocalList::get(bool own_flag) {
	own_thread_executed = own_flag;
	if (pending.size() == 1u && (!own_flag)) {
		pending.back().remove_me();
		return nullptr;
	}
	if (! pending.empty()) {
		Task task = pending.front();
		pending.pop_front();
		executed.push_back(task);
		if (pending.empty()) {
			task.remove_me(); // TODO refact
		}
		return task;
	} else {
		// TODO waiter
		returm nullptr;
	}	
}

LocalList* GlobalList::get() {
	if (! local_lists.empty()) {
		LocalList* list = local_lists.front;
		local_lists.pop_front();
		while ( !) //
	}
}

void Task::execute() {
	if (!m_entry->tainted) {
		return;
	}
	LocalList list;
	list.fill(m_entry);
	list.global_list.add(list);	
	Task parent_task;
	while (parent_task = list.get(true)) {
		parent_task.execute();
	}
	remove_me();
	m_entry.evaluate();
}
