#include <iostream>
#include <vector>
#include "network_util.h"

int main() {
    std::cout << "HedgingEngine starting..." << std::endl;
    if (!init_networking()) return 1;

    socket_t server_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (server_fd == INVALID_SOCK) return 1;

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(8080); // port exposed from docker

    if (bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Bind failed" << std::endl;
        return 1;
    }

    std::cout << "HedgingEngine listening on UDP port 8080..." << std::endl;

    char buffer[1024];
    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);

    while (true) {
        int bytes_received = recvfrom(server_fd, buffer, sizeof(buffer), 0,
            (struct sockaddr*)&client_addr, &client_len);
        if (bytes_received > 0) {
            std::string msg(buffer, bytes_received);
            std::cout << "Received order data: " << msg << std::endl;

            
        }
    }

    CLOSE_SOCKET(server_fd);
    cleanup_networking();
    return 0;
    
}