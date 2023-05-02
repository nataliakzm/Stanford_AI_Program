import collections, sys, os
from logic import *

# Propositional logic
# Sentence: "If it's summer and we're in California, then it doesn't rain."
def formula1a():
    # Predicates to use:
    Summer = Atom('Summer')               # whether it's summer
    California = Atom('California')       # whether we're in California
    Rain = Atom('Rain')                   # whether it's raining
    return Implies(And(Summer, California), Not(Rain))


# Sentence: "It's wet if and only if it is raining or the sprinklers are on."
def formula1b():
    # Predicates to use:
    Rain = Atom('Rain')              # whether it is raining
    Wet = Atom('Wet')                # whether it it wet
    Sprinklers = Atom('Sprinklers')  # whether the sprinklers are on
    return Equiv(Wet, Or(Rain, Sprinklers))


# Sentence: "Either it's day or night (but not both)."
def formula1c():
    # Predicates to use:
    Day = Atom('Day')     # whether it's day
    Night = Atom('Night') # whether it's night
    return Or(And(Day, Not(Night)), And(Not(Day), Night))


# First-order logic
# Sentence: "Every person has a mother."
def formula2a():
    # Predicates to use:
    def Person(x): return Atom('Person', x)        # whether x is a person
    def Mother(x, y): return Atom('Mother', x, y)  # whether x's mother is y
    return Forall('$x', Exists('$y', Implies(Person('$x'), Mother('$x', '$y'))))


# Sentence: "At least one person has no children."
def formula2b():
    # Predicates to use:
    def Person(x): return Atom('Person', x)        # whether x is a person
    def Child(x, y): return Atom('Child', x, y)    # whether x has a child y
    return Exists('$x', Not(Exists('$y', Implies(Person('$x'), Child('$x', '$y')))))


# Return a formula which defines Daughter in terms of Female and Child.
def formula2c():
    # Predicates to use:
    def Female(x): return Atom('Female', x)            # whether x is female
    def Child(x, y): return Atom('Child', x, y)        # whether x has a child y
    def Daughter(x, y): return Atom('Daughter', x, y)  # whether x has a daughter y

    return Forall('$x', Forall('$y', Equiv(
            And(Child('$x', '$y'), Female('$y')),
            Daughter('$x', '$y')
        )))    


# Return a formula which defines Grandmother in terms of Female and Parent.
# Note: It is ok for a person to be her own parent
def formula2d():
    # Predicates to use:
    def Female(x): return Atom('Female', x)                  # whether x is female
    def Parent(x, y): return Atom('Parent', x, y)            # whether x has a parent y
    def Grandmother(x, y): return Atom('Grandmother', x, y)  # whether x has a grandmother y

    return Forall('$x', Forall('$z', Equiv(
        And(Exists('$y', And(Parent('$x', '$y'), Parent('$y', '$z'))), Female('$z')),
        Grandmother('$x', '$z')
    )))    


# Liar puzzle
def liar():
    def TellTruth(x): return Atom('TellTruth', x)
    def CrashedServer(x): return Atom('CrashedServer', x)
    john = Constant('john')
    susan = Constant('susan')
    nicole = Constant('nicole')
    mark = Constant('mark')

    formulas = []
    formulas.append(Equiv(TellTruth(john), Not(CrashedServer(john))))
    formulas.append(Equiv(TellTruth(susan), CrashedServer(nicole)))
    formulas.append(Equiv(TellTruth(mark), CrashedServer(susan)))
    formulas.append(Equiv(TellTruth(nicole), Not(TellTruth(susan))))
    formulas.append(Exists('$x', And(TellTruth('$x'), Forall('$y', Implies(TellTruth('$y'), Equals('$x', '$y'))))))
    formulas.append(Exists('$x', And(CrashedServer('$x'), Forall('$y', Implies(CrashedServer('$y'), Equals('$x', '$y'))))))

    query = CrashedServer('$x')
    return (formulas, query)

# Odd and even integers
def ints():
    def Even(x): return Atom('Even', x)                  # whether x is even
    def Odd(x): return Atom('Odd', x)                    # whether x is odd
    def Successor(x, y): return Atom('Successor', x, y)  # whether x's successor is y
    def Larger(x, y): return Atom('Larger', x, y)        # whether x is larger than y

    formulas = []
    query = None

    formulas.append(Forall('$x', Exists('$y',
        AndList([Successor('$x', '$y'),
            Not(Equals('$x', '$y')),
            Forall('$z', Implies(Successor('$x', '$z'), Equals('$y', '$z')))])
        )))
    formulas.append(Forall('$x', Or(
        And(Even('$x'), Not(Odd('$x'))),
        And(Odd('$x'), Not(Even('$x')))
        )))
    formulas.append(Forall('$x', Forall('$y', Implies(
        And(Even('$x'), Successor('$x', '$y')),
        Odd('$y'))
        )))
    formulas.append(Forall('$x', Forall('$y', Implies(
        And(Odd('$x'), Successor('$x', '$y')),
        Even('$y'))
        )))
    formulas.append(Forall('$x', Forall('$y', Implies(
        Successor('$x', '$y'),
        Larger('$y', '$x')
        ))))
    formulas.append(Forall('$x', Forall('$y', Forall('$z',
       Implies(
            And(Larger('$x', '$y'), Larger('$y', '$z')),
            Larger('$x', '$z')
        )))))    

    query = Forall('$x', Exists('$y', And(Even('$y'), Larger('$y', '$x'))))
    return (formulas, query)

# Semantic parsing (extra)
from nlparser import GrammarRule

def createRule1():
    return GrammarRule('$Clause', ['every', '$Noun', '$Verb', 'some', '$Noun'],
        lambda args: Forall('$x', Forall('$y', Implies(
        And(Atom(args[0].title(), '$x'), Atom(args[1].title(), '$y')),
        Exists('$z', And(Atom(args[2].title(), '$y'), Atom(args[1].title(), args[0].lower(), '$y')))
        ))))    

def createRule2():
    return GrammarRule('$Clause', ['there', 'is', 'some', '$Noun', 'that', 'every', '$Noun', '$Verb'],
        lambda args: Exists('$x',
            Forall('$y', Forall('$z', Implies(
            And(Atom(args[1].title(), '$y'), Atom(args[2].title(), '$z')),
            Atom(args[0].title(), '$x')
            )))))


def createRule3():
    pass