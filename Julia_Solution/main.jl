using CSV
using DataFrames

df = CSV.file("./data.csv")
df = DataFrame(df)
@show df
