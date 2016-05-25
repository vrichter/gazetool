#pragma once

#include <functional>
#include <boost/signals2/signal.hpp>

template<typename Data = std::nullptr_t>
class Signal {
public:
  typedef boost::signals2::connection  Connection;
  typedef Data DataType;

  virtual ~Signal() = default;
  virtual Connection connect(std::function<void (Data)> subscriber) = 0;
  virtual void disconnect(Connection subscriber) = 0;
};

template<typename Data = std::nullptr_t>
class Subject : public Signal<Data> {

private:
    typedef boost::signals2::signal<void (Data)> Signal;
public:
    typedef boost::signals2::connection  Connection;
    typedef Data DataType;
    typedef std::shared_ptr<Subject<Data>> Ptr;

    Subject() = default;
    virtual ~Subject() = default;

    virtual Connection connect(std::function<void (Data)> subscriber) final {
      return m_Signal.connect(subscriber);
    }

    virtual void disconnect(Connection subscriber) final {
      subscriber.disconnect();
    }

    void notify(Data data) {
      m_Signal(data);
    }

private:
    Signal m_Signal;
};
