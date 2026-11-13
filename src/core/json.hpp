#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ce {

// Minimal ordered JSON value, enough for probe reports and plan dumps.
class Json {
 public:
  using Array = std::vector<Json>;
  using Object = std::vector<std::pair<std::string, Json>>;

  Json() = default;
  Json(std::nullptr_t) {}
  Json(bool v) : v_(v) {}
  Json(int v) : v_(int64_t{v}) {}
  Json(unsigned v) : v_(int64_t{v}) {}
  Json(int64_t v) : v_(v) {}
  Json(uint64_t v) : v_(static_cast<int64_t>(v)) {}
  Json(double v) : v_(v) {}
  Json(const char* v) : v_(std::string(v)) {}
  Json(std::string v) : v_(std::move(v)) {}
  Json(std::string_view v) : v_(std::string(v)) {}
  Json(Array v) : v_(std::make_shared<Array>(std::move(v))) {}
  Json(Object v) : v_(std::make_shared<Object>(std::move(v))) {}

  static Json object() { return Json(Object{}); }
  static Json array() { return Json(Array{}); }

  // Object insertion (keeps order). Requires an object.
  Json& set(std::string key, Json value) {
    std::get<std::shared_ptr<Object>>(v_)->emplace_back(std::move(key), std::move(value));
    return *this;
  }
  Json& push(Json value) {
    std::get<std::shared_ptr<Array>>(v_)->push_back(std::move(value));
    return *this;
  }

  bool is_null() const { return std::holds_alternative<std::monostate>(v_); }

  // True iff this is an object whose `key` holds boolean true.
  bool flag(std::string_view key) const {
    auto* m = std::get_if<std::shared_ptr<Object>>(&v_);
    if (!m) return false;
    for (auto& [k, v] : **m)
      if (k == key) {
        auto* b = std::get_if<bool>(&v.v_);
        return b && *b;
      }
    return false;
  }

  std::string dump() const {
    std::string out;
    write(out);
    return out;
  }

 private:
  void write(std::string& o) const {
    struct V {
      std::string& o;
      void operator()(std::monostate) const { o += "null"; }
      void operator()(bool b) const { o += b ? "true" : "false"; }
      void operator()(int64_t i) const { o += std::to_string(i); }
      void operator()(double d) const {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.6g", d);
        o += buf;
      }
      void operator()(const std::string& s) const { escape(o, s); }
      void operator()(const std::shared_ptr<Array>& a) const {
        o += '[';
        for (size_t i = 0; i < a->size(); ++i) {
          if (i) o += ',';
          (*a)[i].write(o);
        }
        o += ']';
      }
      void operator()(const std::shared_ptr<Object>& m) const {
        o += '{';
        for (size_t i = 0; i < m->size(); ++i) {
          if (i) o += ',';
          escape(o, (*m)[i].first);
          o += ':';
          (*m)[i].second.write(o);
        }
        o += '}';
      }
    };
    std::visit(V{o}, v_);
  }

  static void escape(std::string& o, std::string_view s) {
    o += '"';
    for (unsigned char c : s) {
      switch (c) {
        case '"': o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break;
        case '\t': o += "\\t"; break;
        default:
          if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            o += buf;
          } else {
            o += static_cast<char>(c);
          }
      }
    }
    o += '"';
  }

  std::variant<std::monostate, bool, int64_t, double, std::string, std::shared_ptr<Array>,
               std::shared_ptr<Object>>
      v_;
};

}  // namespace ce
