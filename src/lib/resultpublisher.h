#pragma once

#include "gazehyps.h"

class ResultPublisher
{
public:
   typedef std::function<std::unique_ptr<ResultPublisher>(
        const std::string&)
    > FactoryFunction;

    static std::unique_ptr<ResultPublisher> create(const std::string& id, const std::string& param);

    static void registerResultPublisher(const std::string& id, FactoryFunction function, const std::string& description);

    class StaticRegistrar {
    public:
        StaticRegistrar(const std::string& id, FactoryFunction function, const std::string& description){
            try {
                ResultPublisher::registerResultPublisher(id,function,description);
            } catch (std::runtime_error& e) {
                std::cerr << "WARNING: "<< e.what() << std::endl;
            }
        }
    };

    ~ResultPublisher() = default;

    virtual void publish(GazeHypsPtr gazehyps) = 0;

};
