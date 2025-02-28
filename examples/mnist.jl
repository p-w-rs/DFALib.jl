using DFALib
using MLDatasets, OneHotArrays, LinearAlgebra, Random, ProgressMeter, Plots

batch_size = 1024
x_train, y_train = MNIST(split=:train)[:]
x_test, y_test = MNIST(split=:test)[:]
x_train, y_train = Float64.(reshape(x_train, 28 * 28, :)), Float64.(onehotbatch(y_train, 0:9))
x_test, y_test = Float64.(reshape(x_test, 28 * 28, :)), Float64.(onehotbatch(y_test, 0:9))

nn = DFANet(
    Dense(28 * 28, 128, Sigmoid, initW=GlorotNormal, initB=GlorotNormal),
    Dense(128, 64, Sigmoid, initW=GlorotNormal, initB=GlorotNormal),
    Dense(64, 10, Identity, initW=GlorotNormal, initB=GlorotNormal)
)
train_accuracies = Float64[]
test_accuracies = Float64[]
η = 0.001

for epoch in 1:200
    global nn, η
    @showprogress for _ in 1:div(size(y_train, 2), batch_size)
        idxs = randperm(size(x_train, 2))[1:batch_size]
        x, y = x_train[:, idxs], y_train[:, idxs]
        ŷ = softmax(nn(x))
        e = ŷ .- y
        feedback!(nn, e, η)
    end

    train_preds = nn(x_train)
    train_acc = sum(onecold(train_preds) .== onecold(y_train)) / size(y_train, 2)
    push!(train_accuracies, train_acc)

    test_preds = nn(x_test)
    test_acc = sum(onecold(test_preds) .== onecold(y_test)) / size(y_test, 2)
    push!(test_accuracies, test_acc)

    println("Epoch $epoch: Train accuracy = $train_acc, Test accuracy = $test_acc")
    η *= 0.99f0
end

# Plot the results
plot(train_accuracies, label="Train Accuracy", title="Learning Progress", xlabel="Epoch", ylabel="Accuracy")
plot!(test_accuracies, label="Test Accuracy")
savefig("learning_progress.png")
