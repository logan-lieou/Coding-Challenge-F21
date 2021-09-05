### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ ddd8c038-4c76-4d77-b6fa-90cb8df1332c
using Pkg; Pkg.add("AutoMLPipeline");

# ╔═╡ fce7be66-0dfa-11ec-12c1-5f2207106e0c
begin
	using CSV
	using DataFrames
end

# ╔═╡ 1acaecd6-956d-45bb-baf2-53ce9ae777a8
using Flux

# ╔═╡ f4225b3d-fb76-404f-ae10-fe792ec5f39d
df = DataFrame(CSV.File("data.csv"))

# ╔═╡ 6bc0e02b-b8f6-413f-bcd6-253482e8f171
labels = df[:, "sentiment values"]

# ╔═╡ 65522175-6c18-4be1-8567-ec70cb507826
begin
	features = df[:, "body_text"]
	features = cat(features, df[:, "phrase ids"], dims=(2,2))
end

# ╔═╡ b4ca7dd4-dafb-4ab4-b2c2-e47fc57aa4c5
num_rows = 239231

# ╔═╡ 5fd7eb7c-d252-4e7b-905a-1c2a62d26a18
train_split = round(Int, num_rows * .7)

# ╔═╡ 6feffc8f-0001-4969-b2f4-076e90990834
begin
	train_features = features[1:train_split, :]
end

# ╔═╡ 7aab13b9-70fe-4b03-8b05-d778ec1fccf0
train_labels = labels[1:train_split]

# ╔═╡ 1d4e12c4-95c5-42fa-a286-b48c1f954326
test_features = features[train_split:end, :]

# ╔═╡ 6e1c5890-b872-4b04-bde2-ce3c1e2eb12a
test_labels = labels[train_split:end]

# ╔═╡ 689ae96c-ed46-4668-9754-fb04aaf80992
model = Chain(Dense(2, 2), Dense(2, 16), softmax)

# ╔═╡ 88d5e851-3618-4d0e-9294-f9e1c74682be
L(x, y) = Flux.Losses.crossentropy(model(x), y)

# ╔═╡ 84918ac2-edfd-4c06-946d-c0cb7edbfb16
opt = ADAMW(0.02)

# ╔═╡ Cell order:
# ╠═fce7be66-0dfa-11ec-12c1-5f2207106e0c
# ╠═f4225b3d-fb76-404f-ae10-fe792ec5f39d
# ╠═6bc0e02b-b8f6-413f-bcd6-253482e8f171
# ╠═65522175-6c18-4be1-8567-ec70cb507826
# ╠═b4ca7dd4-dafb-4ab4-b2c2-e47fc57aa4c5
# ╠═5fd7eb7c-d252-4e7b-905a-1c2a62d26a18
# ╠═6feffc8f-0001-4969-b2f4-076e90990834
# ╠═7aab13b9-70fe-4b03-8b05-d778ec1fccf0
# ╠═1d4e12c4-95c5-42fa-a286-b48c1f954326
# ╠═6e1c5890-b872-4b04-bde2-ce3c1e2eb12a
# ╠═1acaecd6-956d-45bb-baf2-53ce9ae777a8
# ╠═689ae96c-ed46-4668-9754-fb04aaf80992
# ╠═88d5e851-3618-4d0e-9294-f9e1c74682be
# ╠═84918ac2-edfd-4c06-946d-c0cb7edbfb16
# ╠═ddd8c038-4c76-4d77-b6fa-90cb8df1332c
