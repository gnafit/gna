#include "GNAObject.cc"
#include "TransformationEntry.hh"


class Task {
    Task(TransformationTypes::Entry* entry, LocalList* list) : m_entry(entry), parents(list) {}

    TransformationTypes::Entry *m_entry;
    LocalList* parents;

    void execute_safe();
    void remove_me();
private:
    void execute();
    
};

class LocalList {
    GlobalList* global_list;
//    TransformationTypes::Entry* key_entry;
    Task key_task;
    list<Task> panding;
    list<Task> executed;
    std::atomic<bool> own_thread_executed=false;

    Task* get(bool own_flag = false);

private:
    void fill(TransformationTypes::Entry* entry);	///< Fill this list by transformation parents
};

class GlobalList {
    list<LocalList> local_lists;

    get();
    void add(LocalList);
    void runEntry();
};
