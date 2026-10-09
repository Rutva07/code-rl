#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <sys/stat.h>
#include <time.h>
#include <cerrno>

static void limit(int what, rlim_t soft, rlim_t hard) {
    rlimit value{soft, hard};
    if (setrlimit(what, &value) != 0) { perror("setrlimit"); std::exit(125); }
}

static double ms(const timespec& t) { return t.tv_sec * 1000.0 + t.tv_nsec / 1e6; }
static double ms(const timeval& t) { return t.tv_sec * 1000.0 + t.tv_usec / 1000.0; }

int main(int argc, char** argv) {
    if (argc < 5) return 125;
    const auto seconds = static_cast<rlim_t>(std::ceil(std::atof(argv[1])));
    const auto bytes = static_cast<rlim_t>(std::atol(argv[2])) * 1024 * 1024;
    umask(0077);
    timespec start{}, end{};
    clock_gettime(CLOCK_MONOTONIC, &start);
    const pid_t child = fork();
    if (child < 0) return 125;
    if (child == 0) {
        limit(RLIMIT_CPU, seconds, seconds + 1);
        limit(RLIMIT_AS, bytes, bytes);
        limit(RLIMIT_FSIZE, 4 * 1024 * 1024, 4 * 1024 * 1024);
        limit(RLIMIT_CORE, 0, 0);
        limit(RLIMIT_NOFILE, 64, 64);
        execv(argv[4], argv + 4);
        perror("execv");
        _exit(126);
    }
    int status = 0;
    rusage usage{};
    while (wait4(child, &status, 0, &usage) < 0) {
        if (errno != EINTR) return 125;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    FILE* output = fopen(argv[3], "w");
    if (!output) return 125;
    fprintf(output, "{\"wall_ms\":%.9f,\"user_cpu_ms\":%.9f,\"system_cpu_ms\":%.9f}",
            ms(end)-ms(start), ms(usage.ru_utime), ms(usage.ru_stime));
    fclose(output);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}
