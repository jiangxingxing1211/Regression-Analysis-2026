from simulation import loop, analysis

if __name__ == "__main__":
    df = loop(n_sim=100, n=100)
    analysis(df)