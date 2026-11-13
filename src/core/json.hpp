#pragma once

#include <cctype>
#include <cstdint>
#include <stdexcept>
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
  bool is_object() const { return std::holds_alternative<std::shared_ptr<Object>>(v_); }
  bool is_array() const { return std::holds_alternative<std::shared_ptr<Array>>(v_); }
  bool is_string() const { return std::holds_alternative<std::string>(v_); }
  bool is_number() const { return std::holds_alternative<int64_t>(v_) || std::holds_alternative<double>(v_); }

  const Object& items() const { return *std::get<std::shared_ptr<Object>>(v_); }
  const Array& elements() const { return *std::get<std::shared_ptr<Array>>(v_); }
  const std::string& str() const { return std::get<std::string>(v_); }
  double num() const {
    if (auto* i = std::get_if<int64_t>(&v_)) return static_cast<double>(*i);
    return std::get<double>(v_);
  }
  // Member lookup; null Json if absent or not an object.
  const Json& operator[](std::string_view key) const {
    static const Json null;
    if (!is_object()) return null;
    for (auto& [k, v] : items())
      if (k == key) return v;
    return null;
  }

  // Strict parser for the documents copyengine writes. Throws std::runtime_error.
  static Json parse(std::string_view text);

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
        std::snprintf(buf, sizeof buf, "%.9g", d);
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

  struct Parser;

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

struct Json::Parser {
  std::string_view s;
  size_t i = 0;

  [[noreturn]] void error(const char* what) {
    throw std::runtime_error(std::string("json: ") + what + " at offset " + std::to_string(i));
  }
  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i;
  }
  bool eat(char c) {
    ws();
    if (i < s.size() && s[i] == c) {
      ++i;
      return true;
    }
    return false;
  }
  void expect(char c) {
    if (!eat(c)) error("unexpected character");
  }
  bool lit(std::string_view w) {
    if (s.substr(i, w.size()) == w) {
      i += w.size();
      return true;
    }
    return false;
  }
  std::string string() {
    expect('"');
    std::string out;
    while (i < s.size() && s[i] != '"') {
      char c = s[i++];
      if (c == '\\') {
        if (i >= s.size()) error("bad escape");
        char e = s[i++];
        switch (e) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'u': {
            if (i + 4 > s.size()) error("bad \\u escape");
            unsigned cp = static_cast<unsigned>(std::stoul(std::string(s.substr(i, 4)), nullptr, 16));
            i += 4;
            if (cp < 0x80) out += static_cast<char>(cp);
            else if (cp < 0x800) out += static_cast<char>(0xC0 | (cp >> 6)), out += static_cast<char>(0x80 | (cp & 0x3F));
            else out += static_cast<char>(0xE0 | (cp >> 12)), out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)),
                 out += static_cast<char>(0x80 | (cp & 0x3F));
            break;
          }
          default: out += e;
        }
      } else {
        out += c;
      }
    }
    if (i >= s.size()) error("unterminated string");
    ++i;
    return out;
  }
  Json value() {
    ws();
    if (i >= s.size()) error("unexpected end");
    char c = s[i];
    if (c == '{') {
      ++i;
      Json o = Json::object();
      if (eat('}')) return o;
      do {
        ws();
        std::string k = string();
        expect(':');
        o.set(std::move(k), value());
      } while (eat(','));
      expect('}');
      return o;
    }
    if (c == '[') {
      ++i;
      Json a = Json::array();
      if (eat(']')) return a;
      do a.push(value());
      while (eat(','));
      expect(']');
      return a;
    }
    if (c == '"') return Json(string());
    if (lit("true")) return Json(true);
    if (lit("false")) return Json(false);
    if (lit("null")) return Json(nullptr);
    size_t start = i;
    bool real = false;
    while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '-' || s[i] == '+' ||
                            s[i] == '.' || s[i] == 'e' || s[i] == 'E')) {
      real |= s[i] == '.' || s[i] == 'e' || s[i] == 'E';
      ++i;
    }
    if (start == i) error("unexpected character");
    std::string num(s.substr(start, i - start));
    if (real) return Json(std::stod(num));
    return Json(static_cast<int64_t>(std::stoll(num)));
  }
};

inline Json Json::parse(std::string_view text) {
  Parser p{text};
  Json v = p.value();
  p.ws();
  if (p.i != text.size()) p.error("trailing characters");
  return v;
}

}  // namespace ce
