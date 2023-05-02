import util, model

print('Map coloring example:')
csp = util.create_map_coloring_csp()
alg = model.BacktrackingSearch()
alg.solve(csp)
print(('One of the optimal projects:',  alg.optimalProject))

print('\nWeighted CSP example:')
csp = util.create_weighted_csp()
alg = model.BacktrackingSearch()
alg.solve(csp)
print(('One of the optimal projects:',  alg.optimalProject))
