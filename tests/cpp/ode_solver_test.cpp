#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>
#include "../../src/cpp/luTe.hpp"
#include "../../src/cpp/qr.hpp"
#include "../../src/cpp/qrTe.hpp"
#include "../../src/cpp/qrted.hpp"

TEST(LUTest, A1x2x2) {
    int M=1;
    int N=2;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [LU,indx] = janus::LUTe(A);
    auto x = janus::solveluv(LU, indx, B);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}



TEST(LUTest, A1x100x100) {
    int M=1;
    int N=100;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [LU,indx] = janus::LUTe(A);
    auto x = janus::solveluv(LU, indx, B);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}

TEST(LUTest, A10x100x100) {
    int M=10;
    int N=100;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [LU,indx] = janus::LUTe(A);
    auto x = janus::solveluv(LU, indx, B);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}



TEST(QRTTest, A1x2x2) {
    int N=2;
    torch::Tensor A = torch::rand({N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrt(A);
    auto x = janus::qrtsolvev(qt, r, B);
    auto Ax = torch::mv(A, x);
    EXPECT_TRUE(torch::allclose(Ax, B));
}


TEST(QRTTest, A10x10) {
    int N=10;
    torch::Tensor A = torch::rand({N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrt(A);
    auto x = janus::qrtsolvev(qt, r, B);
    auto Ax = torch::mv(A, x);
    EXPECT_TRUE(torch::allclose(Ax, B));
}

TEST(QRTTest, A1x100x100) {
    int N=100;
    torch::Tensor A = torch::rand({N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrt(A);
    auto x = janus::qrtsolvev(qt, r, B);
    auto Ax = torch::mv(A, x);
    EXPECT_TRUE(torch::allclose(Ax, B));
}







TEST(QRTeTest, A1x2x2) {
    int M=1;
    int N=2;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrte(A);
    auto x = janus::qrtesolvev(qt, r, B);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}

TEST(QRTeTest, A4x2x2) {
    int M=4;
    int N=2;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrte(A);
    auto x = janus::qrtesolvev(qt, r, B);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}

TEST(QRTeTest, A1x100x100) {
    int M=1;
    int N=100;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrte(A);
    auto x = janus::qrtesolvev(qt, r, B);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}

TEST(QRTeTest, A100x10x10) {
    int M=100;
    int N=10;
    torch::Tensor A = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor B = torch::rand({M,N}, dtype(torch::kFloat64));
    auto [qt, r] = janus::qrte(A);
    auto x = janus::qrtesolvev(qt, r, B);
    for (int i=0; i<M; i++){
        auto Ax = torch::mv(A.index({i}), x.index({i}));
        EXPECT_TRUE(torch::allclose(Ax, B.index({i}), 1e-6));
    }
}




TEST(QRTeDTest, A1x2x2) {
    int M=1;
    int N=2;
    torch::Tensor Ar = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor Ad = torch::rand({M,N,N,N}, dtype(torch::kFloat64));
    torch::Tensor Br = torch::rand({M,N}, dtype(torch::kFloat64));
    torch::Tensor Bd = torch::rand({M,N,N}, dtype(torch::kFloat64));
    TensorMatDual A(Ar, Ad);
    TensorDual B(Br, Bd);
    auto [qt, r] = janus::qrted(A);
    auto x = janus::qrtedsolvev(qt, r, B);
    auto Ax = TensorMatDual::einsum("mij, mj->mi", A, x);
    for (int i=0; i<M; i++){
        EXPECT_TRUE(torch::allclose(Ax.r.index({i}), B.r.index({i})));
    }
}

TEST(QRTeDTest, A10x2x2) {
    int M=10;
    int N=2;
    torch::Tensor Ar = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor Ad = torch::rand({M,N,N,N}, dtype(torch::kFloat64));
    torch::Tensor Br = torch::rand({M,N}, dtype(torch::kFloat64));
    torch::Tensor Bd = torch::rand({M,N,N}, dtype(torch::kFloat64));
    TensorMatDual A(Ar, Ad);
    TensorDual B(Br, Bd);
    auto [qt, r] = janus::qrted(A);
    auto x = janus::qrtedsolvev(qt, r, B);
    auto Ax = TensorMatDual::einsum("mij, mj->mi", A, x);
    for (int i=0; i<M; i++){
        EXPECT_TRUE(torch::allclose(Ax.r.index({i}), B.r.index({i})));
    }
}

TEST(QRTeDTest, A10x100x100) {
    int M=10;
    int N=100;
    torch::Tensor Ar = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor Ad = torch::rand({M,N,N,N}, dtype(torch::kFloat64));
    torch::Tensor Br = torch::rand({M,N}, dtype(torch::kFloat64));
    torch::Tensor Bd = torch::rand({M,N,N}, dtype(torch::kFloat64));
    TensorMatDual A(Ar, Ad);
    TensorDual B(Br, Bd);
    auto [qt, r] = janus::qrted(A);
    auto x = janus::qrtedsolvev(qt, r, B);
    auto Ax = TensorMatDual::einsum("mij, mj->mi", A, x);
    for (int i=0; i<M; i++){
        EXPECT_TRUE(torch::allclose(Ax.r.index({i}), B.r.index({i})));
    }
}

TEST(QRTeDTest, A10x200x200) {
    int M=10;
    int N=200;
    torch::Tensor Ar = torch::rand({M,N,N}, dtype(torch::kFloat64));
    torch::Tensor Ad = torch::rand({M,N,N,N}, dtype(torch::kFloat64));
    torch::Tensor Br = torch::rand({M,N}, dtype(torch::kFloat64));
    torch::Tensor Bd = torch::rand({M,N,N}, dtype(torch::kFloat64));
    TensorMatDual A(Ar, Ad);
    TensorDual B(Br, Bd);
    auto [qt, r] = janus::qrted(A);
    auto x = janus::qrtedsolvev(qt, r, B);
    auto Ax = TensorMatDual::einsum("mij, mj->mi", A, x);
    for (int i=0; i<M; i++){
        EXPECT_TRUE(torch::allclose(Ax.r.index({i}), B.r.index({i})));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
