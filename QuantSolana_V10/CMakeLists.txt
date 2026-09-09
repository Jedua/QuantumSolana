cmake_minimum_required(VERSION 3.20)
project(QuantSolana_V10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)

if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wa,-mbig-obj)
endif()

# Search for required Boost, OpenSSL, and Threads
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

# Descargar simdjson automáticamente vía FetchContent
include(FetchContent)
FetchContent_Declare(
    simdjson
    GIT_REPOSITORY https://github.com/simdjson/simdjson.git
    GIT_TAG v0.4.7
    EXCLUDE_FROM_ALL
)

# Esto evita que simdjson fuerce la regla -Werror al compilarse a si mismo
set(SIMDJSON_DEVELOPER_MODE OFF CACHE INTERNAL "")
set(SIMDJSON_ENABLE_THREADS OFF CACHE INTERNAL "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")

FetchContent_MakeAvailable(simdjson)

# Include directories
include_directories(cpp_core/include ${simdjson_SOURCE_DIR}/include ${simdjson_SOURCE_DIR}/src)

# Executable mock_writer
add_executable(mock_writer cpp_core/src/mock_writer.cpp)
target_link_libraries(mock_writer PRIVATE Boost::system Boost::thread OpenSSL::SSL OpenSSL::Crypto Threads::Threads)
if(WIN32)
    target_link_libraries(mock_writer PRIVATE ws2_32 wsock32)
endif()

# Executable binance_ingestion (Ingesta real WebSocket Binance)
add_executable(binance_ingestion 
    cpp_core/src/binance_feed.cpp
    ${simdjson_SOURCE_DIR}/src/simdjson.cpp
)
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(binance_ingestion PRIVATE -O2 -Wa,-mbig-obj -Wno-error)
    target_compile_options(mock_writer PRIVATE -O2 -Wa,-mbig-obj -Wno-error)
    
    # Obliga a GCC a incrustar las librerias estandar dentro del .exe (evita el error libgcc_s_seh-1.dll)
    target_link_options(binance_ingestion PRIVATE "-static")
    target_link_options(mock_writer PRIVATE "-static")
endif()
target_link_libraries(binance_ingestion PRIVATE Boost::system Boost::thread OpenSSL::SSL OpenSSL::Crypto Threads::Threads)

if(WIN32)
    target_link_libraries(binance_ingestion PRIVATE ws2_32 wsock32)
endif()
