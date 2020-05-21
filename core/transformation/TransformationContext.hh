#pragma once

#include <map>
#include <stack>
#include <string>
#include <stdexcept>

namespace TransformationTypes{
    using Attrs      = std::map<std::string,std::string>;
    using AttrsStack = std::map<std::string,std::stack<std::string>>;

    class TransformationContext {
    public:
        static void setAttr(const std::string& key, const std::string& value) {
            context[key]=value;
        }
        static bool hasAttr(const std::string& key) {
            return context.count(key);
        }
        static std::string getAttr(const std::string& key) {
            return context[key];
        }
        static void pushKey(const std::string& key) {
            auto it=context.find(key);
            if( it==context.end() ) {
                context_stack[key];
                return;
            }
            context_stack[key].push(it->second);
        }
        static void popAttr(const std::string& key) {
            auto it=context_stack.find(key);
            if( it==context_stack.end() ) {
                throw std::runtime_error("Unable to pop key from the stack");
            }
            auto& stk=it->second;
            if(stk.empty()){
                context.erase(key);
                context_stack.erase(key);
                return;
            }

            context[key]=stk.top();
            stk.pop();
        }
        static void pushAttr(const std::string& key, const std::string& value) {
            TransformationContext::pushKey(key);
            TransformationContext::setAttr(key, value);
        }
        static const Attrs& getContext() {
            return context;
        }
    private:
        static Attrs context;
        static AttrsStack context_stack;
    };
}
