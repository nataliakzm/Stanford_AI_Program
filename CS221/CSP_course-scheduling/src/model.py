import collections, util, copy

# Hint: Take a look at the CSP class and the CSP examples in util.py
def create_chain_csp(n):
    # same domain for each variable
    domain = [0, 1]
    # name variables as x_1, x_2, ..., x_n
    variables = ['x%d'%i for i in range(1, n+1)]
    csp = util.CSP()
    variables = [f'X{i}' for i in range(1, n + 1)]
    for i in range(n):
        csp.add_variable(variables[i], [0, 1])
    for i in range(n-1):
        csp.add_binary_factor(variables[i], variables[i+1], lambda x, y: x ^ y)  # ^ is XOR 
    return csp

def create_nqueens_csp(n = 8):
    """
    Return an N-Queen problem on the board of size |n| * |n|.
    @param n: number of queens, or the size of one dimension of the board.
    @return csp: A CSP problem with correctly configured factor tables
        such that it can be solved by a weighted CSP solver.
    """
    csp = util.CSP()
    variables = [f'X{i}' for i in range(1, n + 1)]
    domain = list(range(1, n + 1))  # domain of variable is the columns, row is defined by i
    for i in range(n):
        csp.add_variable(variables[i], list(range(1, n + 1)))  
    for i in range(n):
        for j in range(i + 1, n):
            csp.add_binary_factor(variables[i], variables[j], lambda x, y: x != y and abs(x - y) != j - i)  # check for same col and diagonal
    return csp

# A backtracking algorithm that solves weighted CSP.
class BacktrackingSearch():

    def reset_results(self):
        """
        This function resets the statistics of the different aspects of the
        CSP solver.
        """
        # Keep track of the best project and weight found.
        self.optimalProject = {}
        self.optimalWeight = 0

        # Keep track of the number of optimal projects and projects. These
        # two values should be identical when the CSP is unweighted or only has binary
        # weights.
        self.numOptimalProjects = 0
        self.numProjects = 0

        # Keep track of the number of times backtrack() gets called.
        self.numOperations = 0

        # Keep track of the number of operations to get to the very first successful
        # project (doesn't have to be optimal).
        self.firstProjectNumOperations = 0

        # List of all solutions found.
        self.allProjects = []

    def print_stats(self):
        """
        Prints a message summarizing the outcome of the solver.
        """
        if self.optimalProject:
            print(("Found %d optimal projects with weight %f in %d operations" % \
                (self.numOptimalProjects, self.optimalWeight, self.numOperations)))
            print(("First project took %d operations" % self.firstProjectNumOperations))
        else:
            print("No solution was found.")

    def get_delta_weight(self, project, var, val):
        """
        Given a CSP, a partial project, and a proposed new value for a variable,
        return the change of weights after assigning the variable with the proposed
        value.

        @param project: A dictionary of current project. Unassigned variables
            do not have entries, while an assigned variable has the assigned value
            as value in dictionary. e.g. if the domain of the variable A is [5,6],
            and 6 was assigned to it, then project[A] == 6.
        @param var: name of an unassigned variable.
        @param val: the proposed value.

        @return w: Change in weights as a result of the proposed project. This
            will be used as a multiplier on the current weight.
        """
        assert var not in project
        w = 1.0
        if self.csp.unaryFactors[var]:
            w *= self.csp.unaryFactors[var][val]
            if w == 0: return w
        for var2, factor in list(self.csp.binaryFactors[var].items()):
            if var2 not in project: continue  # Not assigned yet
            w *= factor[val][project[var2]]
            if w == 0: return w
        return w

    def solve(self, csp, mcv = False, ac3 = False):
        """
        Solves the given weighted CSP using heuristics as specified in the
        parameter.

        @param csp: A weighted CSP.
        @param mcv: When enabled, Most Constrained Variable heuristics is used.
        @param ac3: When enabled, AC-3 will be used after each project of an
            variable is made.
        """
        # CSP to be solved.
        self.csp = csp

        # Set the search heuristics requested asked.
        self.mcv = mcv
        self.ac3 = ac3

        # Reset solutions from previous search.
        self.reset_results()

        # The dictionary of domains of every variable in the CSP.
        self.domains = {var: list(self.csp.values[var]) for var in self.csp.variables}

        # Perform backtracking search.
        self.backtrack({}, 0, 1)

        # Print summary of solutions.
        self.print_stats()

    def backtrack(self, project, numAssigned, weight):
        """
        Perform the back-tracking algorithms to find all possible solutions to
        the CSP.

        @param project: A dictionary of current project. Unassigned variables
            do not have entries, while an assigned variable has the assigned value
            as value in dictionary. e.g. if the domain of the variable A is [5,6],
            and 6 was assigned to it, then project[A] == 6.
        @param numAssigned: Number of currently assigned variables
        @param weight: The weight of the current partial project.
        """

        self.numOperations += 1
        assert weight > 0
        if numAssigned == self.csp.numVars:
            # A satisfiable solution have been found. Update the statistics.
            self.numProjects += 1
            newProject = {}
            for var in self.csp.variables:
                newProject[var] = project[var]
            self.allProjects.append(newProject)

            if len(self.optimalProject) == 0 or weight >= self.optimalWeight:
                if weight == self.optimalWeight:
                    self.numOptimalProjects += 1
                else:
                    self.numOptimalProjects = 1
                self.optimalWeight = weight

                self.optimalProject = newProject
                if self.firstProjectNumOperations == 0:
                    self.firstProjectNumOperations = self.numOperations
            return

        # Select the next variable to be assigned.
        var = self.get_unassigned_variable(project)
        # Get an ordering of the values.
        ordered_values = self.domains[var]

        # Continue the backtracking recursion using |var| and |ordered_values|.
        if not self.ac3:
            # When arc consistency check is not enabled.
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(project, var, val)
                if deltaWeight > 0:
                    project[var] = val
                    self.backtrack(project, numAssigned + 1, weight * deltaWeight)
                    del project[var]
        else:
            # Arc consistency check is enabled.
            for val in ordered_values:
                deltaWeight = self.get_delta_weight(project, var, val)
                if deltaWeight > 0:
                    project[var] = val
                    # create a deep copy of domains as we are going to look
                    # ahead and change domain values
                    localCopy = copy.deepcopy(self.domains)
                    # fix value for the selected variable so that hopefully we
                    # can eliminate values for other variables
                    self.domains[var] = [val]

                    # enforce arc consistency
                    self.arc_consistency_check(var)
                    self.backtrack(project, numAssigned + 1, weight * deltaWeight)
                    # restore the previous domains
                    self.domains = localCopy
                    del project[var]

    def get_unassigned_variable(self, project):
        """
        Given a partial project, return a currently unassigned variable.
        @param project: A dictionary of current project.
        @return var: a currently unassigned variable.
        """
        if not self.mcv:
            # Select a variable without any heuristics.
            for var in self.csp.variables:
                if var not in project: return var
        else:
            mcv = ''
            min_num_domain_values = float('inf')
            for var in self.csp.variables:
                if var not in project:
                    num_domain_values = 0  # counter for number of consistent values
                    for v in self.domains[var]:
                        if self.get_delta_weight(project, var, v) > 0:
                            num_domain_values += 1
                    if num_domain_values < min_num_domain_values:
                        min_num_domain_values = num_domain_values
                        mcv = var
            return mcv
        

    def arc_consistency_check(self, var):
        """
        Perform the AC-3 algorithm. The goal is to reduce the size of the
        domain values for the unassigned variables based on arc consistency.
        @param var: The variable whose value has just been set.
        """
        def remove_inconsistent_values(var1, var2):
            removed = False
            # the binary factor must exist because we add var1 from var2's neighbor
            factor = self.csp.binaryFactors[var1][var2]
            for val1 in list(self.domains[var1]):
                # Note: in our implementation, it's actually unnecessary to check unary factors,
                #       because in get_delta_weight() unary factors are always checked.
                if (self.csp.unaryFactors[var1] and self.csp.unaryFactors[var1][val1] == 0) or \
                    all(factor[val1][val2] == 0 for val2 in self.domains[var2]):
                    self.domains[var1].remove(val1)
                    removed = True
            return removed

        queue = [(var2, var) for var2 in self.csp.get_neighbor_vars(var)]
        while len(queue) > 0:
            var1, var2 = queue.pop(0)
            if remove_inconsistent_values(var1, var2):
                for var3 in self.csp.get_neighbor_vars(var1):
                    queue.append((var3, var1))

def get_sum_variable(csp, name, variables, maxSum):
    """
    Given a list of |variables| each with non-negative integer domains,
    returns the name of a new variable with domain range(0, maxSum+1), such that
    it's consistent with the value |n| iff the projects for |variables|
    sums to |n|.

    @param name: Prefix of all the variables that are going to be added.
        Can be any hashable objects. For every variable |var| added in this
        function, it's recommended to use a naming strategy such as
        ('sum', |name|, |var|) to avoid conflicts with other variable names.
    @param variables: A list of variables that are already in the CSP that
        have non-negative integer values as its domain.
    @param maxSum: An integer indicating the maximum sum value allowed. 

    @return result: The name of a newly created variable with domain range
        [0, maxSum] such that it's consistent with an project of |n|
        iff the project of |variables| sums to |n|.
    """
    result = ('sum', name, 'aggregated')
    if len(variables) == 0:  # no input variable, check is sum is 0
        csp.add_variable(result, [0])
        return result
    csp.add_variable(result, range(1, maxSum + 1))

    for i, X_i in enumerate(variables):
        A_i = ('sum', name, i)
        if i == 0:
            csp.add_variable(A_i, [(0, k) for k in range(maxSum + 1)])
            csp.add_unary_factor(A_i, lambda b: b[0] == 0)
        else:
            csp.add_variable(A_i, [(j, k) for j in range(maxSum + 1) for k in range(maxSum + 1)])
            # consistency between A_{i-1} and A_i
            csp.add_binary_factor(('sum', name, i - 1), A_i, lambda x, y: x[1] == y[0])
        csp.add_binary_factor(A_i, X_i, lambda x, y: x[1] == (x[0] + y))
    csp.add_binary_factor(A_i, result, lambda x, y: x[1] == y)
    return result  


# importing get_or_variable helper function from util
get_or_variable = util.get_or_variable

# A class providing methods to generate CSP that can solve the course scheduling
# problem.
class SchedulingCSPConstructor():

    def __init__(self, bulletin, profile):
        """
        Saves the necessary data.

        @param bulletin: Stanford Bulletin that provides a list of courses
        @param profile: A student's profile and requests
        """
        self.bulletin = bulletin
        self.profile = profile

    def add_variables(self, csp):
        """
        Adding the variables into the CSP. Each variable, (request, quarter),
        can take on the value of one of the courses requested in request or None.
        For instance, for quarter='Aut2013', and a request object, request, generated
        from 'CS221 or CS246', then (request, quarter) should have the domain values
        ['CS221', 'CS246', None]. Conceptually, if var is assigned 'CS221'
        then it means we are taking 'CS221' in 'Aut2013'. If it's None, then
        we not taking either of them in 'Aut2013'.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for request in self.profile.requests:
            for quarter in self.profile.quarters:
                csp.add_variable((request, quarter), request.cids + [None])

    def add_bulletin_constraints(self, csp):
        """
        Add the constraints that a course can only be taken if it's offered in
        that quarter.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for request in self.profile.requests:
            for quarter in self.profile.quarters:
                csp.add_unary_factor((request, quarter), \
                    lambda cid: cid is None or \
                        self.bulletin.courses[cid].is_offered_in(quarter))

    def add_norepeating_constraints(self, csp):
        """
        No course can be repeated. Coupling with our problem's constraint that
        only one of a group of requested course can be taken, this implies that
        every request can only be satisfied in at most one quarter.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for request in self.profile.requests:
            for quarter1 in self.profile.quarters:
                for quarter2 in self.profile.quarters:
                    if quarter1 == quarter2: continue
                    csp.add_binary_factor((request, quarter1), (request, quarter2), \
                        lambda cid1, cid2: cid1 is None or cid2 is None)

    def get_basic_csp(self):
        """
        Return a CSP that only enforces the basic constraints that a course can
        only be taken when it's offered and that a request can only be satisfied
        in at most one quarter.

        @return csp: A CSP where basic variables and constraints are added.
        """
        csp = util.CSP()
        self.add_variables(csp)
        self.add_bulletin_constraints(csp)
        self.add_norepeating_constraints(csp)
        return csp

    def add_quarter_constraints(self, csp):
        """
        If the profile explicitly wants a request to be satisfied in some given
        quarters, e.g. Aut2013, then add constraints to not allow that request to
        be satisfied in any other quarter. If a request doesn't specify the 
        quarter(s), do nothing.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for request in self.profile.requests:
            for quarter in self.profile.quarters:
                if quarter not in request.quarters:
                    csp.add_unary_factor((request, quarter), \
                        lambda cid: cid is None)       
    

    def add_request_weights(self, csp):
        """
        Incorporate weights into the CSP. By default, a request has a weight
        value of 1 (already configured in Request). A unsatisfied request
        should also have a weight value of 1.

        @param csp: The CSP where the additional constraints will be added to.
        """
        for request in self.profile.requests:
            for quarter in self.profile.quarters:
                csp.add_unary_factor((request, quarter), \
                    lambda cid: request.weight if cid != None else 1.0)

    def add_prereq_constraints(self, csp):
        """
        Adding constraints to enforce prerequisite. A course can have multiple
        prerequisites. We assume that *all courses in req.prereqs are
        being requested*. Note that if the parser inferred that one of a
        requested course has additional prerequisites that are also being
        requested, these courses will be added to req.prereqs. We will be notified
        with a message when this happens. Also note that req.prereqs apply to every
        single course in req.cids. If a course C has prerequisite A that is requested
        together with another course B (i.e. a request of 'A or B'), then taking B does
        not count as satisfying the prerequisite of C. We cannot take a course
        in a quarter unless all of its prerequisites have been taken *before* that
        quarter. 

        @param csp: The CSP where the additional constraints will be added to.
        """
        # Iterate over all request courses
        for req in self.profile.requests:
            if len(req.prereqs) == 0: continue
            # Iterate over all possible quarters
            for quarter_i, quarter in enumerate(self.profile.quarters):
                # Iterate over all prerequisites of this request
                for pre_cid in req.prereqs:
                    # Find the request with this prerequisite
                    for pre_req in self.profile.requests:
                        if pre_cid not in pre_req.cids: continue
                        # Make sure this prerequisite is taken before the requested course(s)
                        prereq_vars = [(pre_req, q) \
                            for i, q in enumerate(self.profile.quarters) if i < quarter_i]
                        v = (req, quarter)
                        orVar = get_or_variable(csp, (v, pre_cid), prereq_vars, pre_cid)
                        # Note this constraint is enforced only when the course is taken
                        # in `quarter` (that's why we test `not val`)
                        csp.add_binary_factor(orVar, v, lambda o, val: not val or o)

    def add_unit_constraints(self, csp):
        """
        Add constraint to the CSP to ensure that the total number of units are
        within profile.minUnits/maxUnits, inclusively. The allowed range for
        each course can be obtained from bulletin.courses[cid].minUnits/maxUnits.
        For a request 'A or B', if we choose to take A, then we must use a unit
        number that's within the range of A. In order for our solution extractor to
        obtain the number of units, for every requested course, we must have
        a variable named (courseId, quarter) (e.g. ('CS221', 'Aut2013')) and
        its assigned value is the number of units.

        @param csp: The CSP where the additional constraints will be added to.
        """    
        for quarter in self.profile.quarters:
            new_vars = []
            for request in self.profile.requests:
                for cid in request.cids:
                    var = (cid, quarter)
                    minVal = self.bulletin.courses[cid].minUnits
                    maxVal = self.bulletin.courses[cid].maxUnits
                    csp.add_variable(var, list(range(minVal, maxVal + 1)) + [0])  #  0 if course not taken in that quarter
                    new_vars.append(var)
                    csp.add_binary_factor((request, quarter), var, lambda request_cid, course_unit: course_unit > 0 if request_cid == cid else course_unit == 0)
            quarter_sum = get_sum_variable(csp, quarter, new_vars, self.profile.maxUnits)
            csp.add_unary_factor(quarter_sum, lambda x: self.profile.minUnits <= x <= self.profile.maxUnits)    
    

    def add_all_additional_constraints(self, csp):
        """
        Add all additional constraints to the CSP.
        @param csp: The CSP where the additional constraints will be added to.
        """
        self.add_quarter_constraints(csp)
        self.add_request_weights(csp)
        self.add_prereq_constraints(csp)
        self.add_unit_constraints(csp)
