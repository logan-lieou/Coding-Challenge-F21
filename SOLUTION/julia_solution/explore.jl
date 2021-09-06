### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ ce71b5fe-0edc-11ec-1f22-3d5fa7013b26
using CSV, DataFrames

# ╔═╡ 1ad61310-ce56-47d4-8073-386d95eb3088
df = DataFrame(CSV.File("data/tweets.csv"))

# ╔═╡ 337dc858-a303-437e-94a3-5ded34acecf8
describe(df, :nmissing)

# ╔═╡ 3377f0b8-d782-43a1-88c8-e89435d4ae6a
df.text

# ╔═╡ f02f0089-f13b-4168-9512-e3012ce9bce3
df.airline_sentiment

# ╔═╡ 00cec535-bcda-4259-88c1-e91b898fc402
begin
	ndf = DataFrame(CSV.File("data/data.csv"))
	ndf = ndf[:, ["body_text", "sentiment values"]]
	rename!(ndf, ["text", "sentiment"])
end

# ╔═╡ Cell order:
# ╠═ce71b5fe-0edc-11ec-1f22-3d5fa7013b26
# ╠═1ad61310-ce56-47d4-8073-386d95eb3088
# ╠═337dc858-a303-437e-94a3-5ded34acecf8
# ╠═3377f0b8-d782-43a1-88c8-e89435d4ae6a
# ╠═f02f0089-f13b-4168-9512-e3012ce9bce3
# ╠═00cec535-bcda-4259-88c1-e91b898fc402
