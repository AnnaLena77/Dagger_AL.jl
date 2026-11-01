include("tileflow_defs.jl")
@testset "TileFlow (one process)" begin
    tileflow_checks(0, 1)
end
