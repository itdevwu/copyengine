#include "copyengine/copyengine.hpp"

#include "copyengine/config.hpp"

namespace ce {

Version version() noexcept { return {CE_VERSION_MAJOR, CE_VERSION_MINOR, CE_VERSION_PATCH}; }

std::string_view version_string() noexcept { return CE_VERSION_STRING; }

}  // namespace ce
