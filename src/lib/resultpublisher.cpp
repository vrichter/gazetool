#include <stdexcept>

#include "resultpublisher.h"

namespace { // do not clutter namespace
  struct ResultPublisherRegister {
    std::mutex mutex;
    std::map<std::string,std::pair<ResultPublisher::FactoryFunction,std::string>> map;

    void add(const std::string& id, ResultPublisher::FactoryFunction function,const std::string& description){
      std::lock_guard<std::mutex> lock(mutex);
      auto it = map.find(id);
      if(it == map.end()){
        map[id] = std::make_pair(function,description);
      } else {
        throw std::runtime_error("ResultPublisher '" + id + "' already registered");
      }
    }

    ResultPublisher::FactoryFunction get(const std::string& id){
      std::lock_guard<std::mutex> lock(mutex);
      auto it = map.find(id);
      if(it != map.end()){
        return it->second.first;
      } else {
        throw std::runtime_error("unknown ResultPublisher '" + id + "'.");
      }
    }

    std::string description(){
      std::lock_guard<std::mutex> lock(mutex);
      uint length = 0;
      for(auto it : map){
        if (it.first.size() > length){
          length = it.first.size();
        }
      }
      std::stringstream builder;
      builder << "The following result publishers are available:\n";
      for(auto it : map){
        builder << "\t" << it.first;
        for(uint i = it.first.size(); i < length; ++i){
          builder << " ";
        }
        builder << "  arg    " << it.second.second << "\n";
      }
      return builder.str();
    }
  };
  static ResultPublisherRegister RPREG;
} // anonymous namespace

void ResultPublisher::registerResultPublisher(const std::string& id, ResultPublisher::FactoryFunction function,
                                          const std::string& description)
{
    RPREG.add(id,function,description);
}

std::unique_ptr<ResultPublisher>
ResultPublisher::create(const std::string& type, const std::string& params)
{
  auto func = RPREG.get(type);
  return func(params);
}

namespace {
static ResultPublisher::StaticRegistrar list(
    "list",
    [](const std::string& params)
    -> std::unique_ptr<ResultPublisher>
    {
      std::cout << RPREG.description() << std::endl;
      throw std::runtime_error("List output cannot be used as publisher.");
    },
    "arg = _ignored_. This output only prints known outputs and throws."
);
}

