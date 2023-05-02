import util, model, sys

if len(sys.argv) < 2:
    print(("Usage: %s <profile file (e.g., profile2d.txt)>" % sys.argv[0]))
    sys.exit(1)

profilePath = sys.argv[1]
bulletin = util.CourseBulletin('courses.json')
profile = util.Profile(bulletin, profilePath)
profile.print_info()
cspConstructor = model.SchedulingCSPConstructor(bulletin, profile)
csp = cspConstructor.get_basic_csp()
cspConstructor.add_all_additional_constraints(csp)

alg = model.BacktrackingSearch()
alg.solve(csp, mcv = True, ac3 = True)
if alg.optimalProject:
    print((alg.optimalWeight))
    for key, value in list(alg.optimalProject.items()):
        print((key, '=', value))

if alg.numOptimalProjects > 0:
    solution = util.extract_course_scheduling_solution(profile, alg.optimalProject)
    util.print_course_scheduling_solution(solution)
