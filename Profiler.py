
import numpy as np

if __name__ == "__main__":
    import cProfile
    
    cProfile.run("main()", "output.dat")
    import Racetrack
    import pstats
    from pstats import SortKey
    
    with open("output_time.txt", "w") as f:
        p = pstats.Stats("output.dat", stream =f)
        p.sort_stats("time").print_stats()
    with open("output_calls.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()
