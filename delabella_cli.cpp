#include <cstddef>
#include <array>
#include "delabella/delabella.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input> <output>\n";
        return 1;
    }

    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "Failed to open input file\n";
        return 2;
    }

    std::size_t point_count = 0;
    int enforce_flag = 0;
    if (!(in >> point_count >> enforce_flag)) {
        std::cerr << "Invalid header\n";
        return 3;
    }

    struct Point { double x; double y; };
    std::vector<Point> pts(point_count);
    for (std::size_t i = 0; i < point_count; ++i) {
        if (!(in >> pts[i].x >> pts[i].y)) {
            std::cerr << "Invalid point entry\n";
            return 4;
        }
    }

    std::size_t edge_count = 0;
    if (!(in >> edge_count)) {
        std::cerr << "Invalid edge count\n";
        return 5;
    }

    std::vector<int> edge_a(edge_count), edge_b(edge_count);
    for (std::size_t i = 0; i < edge_count; ++i) {
        int a = 0, b = 0;
        if (!(in >> a >> b)) {
            std::cerr << "Invalid edge entry\n";
            return 6;
        }
        edge_a[i] = a;
        edge_b[i] = b;
    }
    in.close();

    using DelaBella = IDelaBella2<double, int>;
    DelaBella* idb = DelaBella::Create();
    if (!idb) {
        std::cerr << "Failed to create DelaBella instance\n";
        return 7;
    }

    const int num_points = static_cast<int>(point_count);
    Point* base = pts.data();
    int triangulate_result = idb->Triangulate(num_points, &base->x, &base->y, sizeof(Point), -1);
    if (triangulate_result <= 0) {
        std::cerr << "Triangulation failed or degenerate input\n";
        idb->Destroy();
        return 8;
    }

    const bool enforce = enforce_flag != 0 && edge_count > 0;
    if (enforce) {
        const int num_edges = static_cast<int>(edge_count);
        idb->ConstrainEdges(num_edges, edge_a.data(), edge_b.data(), sizeof(int));
        idb->FloodFill(false, nullptr, 0);
    }

    using Simplex = DelaBella::Simplex;
    using Vertex = DelaBella::Vertex;

    const Simplex* simplex = idb->GetFirstDelaunaySimplex();
    std::unordered_map<const Vertex*, int> vertex_map;
    vertex_map.reserve(static_cast<std::size_t>(idb->GetNumPolygons()) * 3);
    std::vector<std::array<int, 3>> triangles;
    triangles.reserve(static_cast<std::size_t>(idb->GetNumPolygons()));
    std::vector<std::array<double, 2>> vertices;

    while (simplex) {
        if (!simplex->IsDelaunay()) {
            simplex = simplex->next;
            continue;
        }
        if (enforce && ((simplex->flags & 0b01000000) == 0)) {
            simplex = simplex->next;
            continue;
        }

        std::array<int, 3> tri_idx{};
        for (int i = 0; i < 3; ++i) {
            const Vertex* v = simplex->v[i];
            auto it = vertex_map.find(v);
            if (it == vertex_map.end()) {
                const int new_index = static_cast<int>(vertices.size());
                vertex_map.emplace(v, new_index);
                vertices.push_back({v->x, v->y});
                tri_idx[i] = new_index;
            } else {
                tri_idx[i] = it->second;
            }
        }
        triangles.push_back(tri_idx);
        simplex = simplex->next;
    }

    std::vector<int> input_to_output(point_count, -1);
    for (std::size_t i = 0; i < point_count; ++i) {
        const Vertex* v = idb->GetVertexByIndex(static_cast<int>(i));
        if (!v) {
            continue;
        }
        auto it = vertex_map.find(v);
        if (it != vertex_map.end()) {
            input_to_output[i] = it->second;
        }
    }

    std::vector<std::array<int, 2>> lines;
    if (enforce) {
        lines.reserve(edge_count);
        for (std::size_t i = 0; i < edge_count; ++i) {
            const int a = input_to_output[static_cast<std::size_t>(edge_a[i])];
            const int b = input_to_output[static_cast<std::size_t>(edge_b[i])];
            if (a >= 0 && b >= 0 && a != b) {
                lines.push_back({a, b});
            }
        }
    }

    idb->Destroy();

    std::ofstream out(argv[2]);
    if (!out) {
        std::cerr << "Failed to open output file\n";
        return 9;
    }

    out << std::setprecision(17);
    out << vertices.size() << "\n";
    for (const auto& p : vertices) {
        out << p[0] << ' ' << p[1] << "\n";
    }
    out << triangles.size() << "\n";
    for (const auto& tri : triangles) {
        out << tri[0] << ' ' << tri[1] << ' ' << tri[2] << "\n";
    }
    out << lines.size() << "\n";
    for (const auto& seg : lines) {
        out << seg[0] << ' ' << seg[1] << "\n";
    }

    out.close();
    return 0;
}
