"""COS 380 course self-check utilities.

Instructor-provided file. Students should not edit this file unless instructed.
The visible checks are intended for debugging and submission readiness only;
official grading may use additional instructor-only tests.
"""
from pathlib import Path
import importlib.util
import numpy as np


def _ok(msg):
    print(f"✓ {msg}")


def _bad(msg, detail=None):
    print(f"✗ {msg}")
    if detail:
        print(f"    {detail}")


def _callable(ns, name):
    obj = ns.get(name)
    if callable(obj):
        _ok(f"{name} found")
        return obj
    _bad(f"{name} found", "Required function is missing or not callable.")
    return None


def course_setup_check(course_root=None):
    """Check that the COS 380 workspace is ready for labs."""
    root = Path(course_root or Path.cwd()).resolve()
    # If called from labs/labXX, walk upward to the folder containing nlp_toolkit.
    candidates = [root] + list(root.parents)
    root = next((p for p in candidates if (p / 'nlp_toolkit').exists() and (p / 'course_tools').exists()), root)

    print('=' * 62)
    print('COS 380 COURSE SETUP CHECK')
    print('=' * 62)

    checks = []
    required_dirs = ['data', 'labs', 'nlp_toolkit', 'course_tools', 'project', 'resources']
    for name in required_dirs:
        exists = (root / name).exists()
        checks.append(exists)
        (_ok if exists else _bad)(f"{name}/ folder found")

    required_files = [
        root / 'nlp_toolkit' / '__init__.py',
        root / 'course_tools' / '__init__.py',
        root / 'course_tools' / 'cos380_checks.py',
        root / 'README.md',
    ]
    for path in required_files:
        exists = path.exists()
        checks.append(exists)
        (_ok if exists else _bad)(f"{path.relative_to(root)} found")

    packages = ['numpy', 'pandas', 'nltk', 'sklearn', 'matplotlib']
    for pkg in packages:
        exists = importlib.util.find_spec(pkg) is not None
        checks.append(exists)
        (_ok if exists else _bad)(f"Python package '{pkg}' available")

    print('-' * 62)
    if all(checks):
        print('Setup status: READY')
    else:
        print('Setup status: NEEDS ATTENTION')
    print('Course root:', root)
    print('=' * 62)
    return all(checks)


def validate_lab1(namespace):
    print('=' * 62)
    print('COS 380 LAB 1 - VISIBLE SELF-CHECK')
    print('=' * 62)

    tests = 0
    passed = 0

    required = [
        'lowercase',
        'remove_urls',
        'remove_mentions',
        'handle_hashtags',
        'remove_punctuation',
        'normalize_whitespace',
        'tokenize',
        'remove_stopwords',
        'stem_tokens',
        'get_wordnet_pos',
        'lemmatize_tokens',
    ]

    functions = {}
    for name in required:
        tests += 1
        fn = _callable(namespace, name)
        if fn:
            _ok(f'{name} exists and is callable')
            passed += 1
            functions[name] = fn

    behavior_tests = [
        ('lowercase basic behavior',
         lambda: functions['lowercase']('HeLLo') == 'hello'),
        ('remove_urls removes URL text',
         lambda: 'http' not in functions['remove_urls']('visit https://example.com now').lower()),
        ('remove_mentions removes @mention',
         lambda: '@united' not in functions['remove_mentions']('@United please help').lower()),
        ('normalize_whitespace collapses repeated spaces',
         lambda: '  ' not in functions['normalize_whitespace']('too   many   spaces')),
        ('tokenize returns a non-empty token sequence',
         lambda: isinstance(functions['tokenize']("I can't wait 4 hours"), (list, tuple))
                 and len(functions['tokenize']("I can't wait 4 hours")) > 0),
        ('remove_stopwords supports keep_negation',
         lambda: 'not' in [str(x).lower() for x in
                           functions['remove_stopwords'](['i','do','not','like','this'],
                                                         keep_negation=True)]),
        ('stem_tokens returns one output per input token',
         lambda: len(functions['stem_tokens'](['running','studies'])) == 2),
        ('get_wordnet_pos returns a WordNet POS value',
         lambda: functions['get_wordnet_pos']('VBG') is not None),
        ('lemmatize_tokens returns one output per input token',
         lambda: len(functions['lemmatize_tokens'](['cars','studies'])) == 2),
    ]

    for label, test_fn in behavior_tests:
        tests += 1
        try:
            ok = bool(test_fn())
            (_ok if ok else _bad)(label)
            passed += int(ok)
        except KeyError:
            _bad(label, 'required function was not available')
        except Exception as e:
            _bad(label, repr(e))

    print(f'Visible checks passed: {passed}/{tests}')
    return passed, tests


def validate_lab2(namespace):
    """Visible self-checks for Lab 2: Bag-of-Words vectorization.

    The scored checks focus on the four reusable functions in
    nlp_toolkit/vectorizer.py. If the Task 7 scikit-learn variables are
    present in the notebook namespace, the checker also prints an unscored
    comparison diagnostic.
    """
    print('=' * 62)
    print('COS 380 LAB 2 - VISIBLE SELF-CHECK')
    print('=' * 62)

    passed = 0
    total = 6

    fn_vocab = _callable(namespace, 'build_vocabulary')
    if fn_vocab:
        try:
            vocab = fn_vocab([['cat', 'sat'], ['cat', 'slept']])
            ok = list(vocab) == ['cat', 'sat', 'slept']
            (_ok if ok else _bad)(
                'build_vocabulary returns unique terms in deterministic order'
            )
            passed += int(ok)
        except Exception as e:
            _bad('build_vocabulary runs', repr(e))

    fn_map = _callable(namespace, 'create_word_to_index')
    if fn_map:
        try:
            mapping = fn_map(['cat', 'sat', 'slept'])
            ok = (
                isinstance(mapping, dict)
                and mapping == {'cat': 0, 'sat': 1, 'slept': 2}
            )
            (_ok if ok else _bad)(
                'create_word_to_index preserves vocabulary order'
            )
            passed += int(ok)
        except Exception as e:
            _bad('create_word_to_index runs', repr(e))

    fn_doc = _callable(namespace, 'vectorize_document')
    if fn_doc:
        try:
            mapping = {'cat': 0, 'sat': 1}
            vector = np.asarray(fn_doc(['cat', 'cat', 'sat'], mapping))
            ok = vector.ndim == 1 and vector.tolist() == [2, 1]
            (_ok if ok else _bad)(
                'vectorize_document counts terms in the correct positions'
            )
            passed += int(ok)
        except Exception as e:
            _bad('vectorize_document counting test runs', repr(e))

        try:
            mapping = {'cat': 0, 'sat': 1}
            vector = np.asarray(
                fn_doc(['cat', 'unknown', 'sat', 'unknown'], mapping)
            )
            ok = vector.tolist() == [1, 1]
            (_ok if ok else _bad)(
                'vectorize_document safely ignores tokens outside the vocabulary'
            )
            passed += int(ok)
        except Exception as e:
            _bad('vectorize_document unknown-token test runs', repr(e))

    fn_corpus = _callable(namespace, 'vectorize_corpus')
    if fn_corpus:
        try:
            mapping = {'cat': 0, 'sat': 1}
            X = np.asarray(
                fn_corpus([['cat', 'cat'], ['sat'], ['cat', 'sat']], mapping)
            )
            ok = X.shape == (3, 2)
            (_ok if ok else _bad)(
                'vectorize_corpus creates one row per document and one column per vocabulary term'
            )
            passed += int(ok)
        except Exception as e:
            _bad('vectorize_corpus shape test runs', repr(e))

        try:
            mapping = {'cat': 0, 'sat': 1}
            X = np.asarray(
                fn_corpus([['cat', 'cat'], ['sat'], ['cat', 'sat']], mapping)
            )
            expected = np.array([
                [2, 0],
                [0, 1],
                [1, 1],
            ])
            ok = np.array_equal(X, expected)
            (_ok if ok else _bad)(
                'vectorize_corpus reuses the same word-to-index mapping for every document'
            )
            passed += int(ok)
        except Exception as e:
            _bad('vectorize_corpus count test runs', repr(e))

    print('-' * 62)
    print(f'Visible checks passed: {passed}/{total}')

    required_task7 = [
        'vocabulary',
        'word_to_index',
        'sk_vectorizer',
        'X_manual',
        'X_sklearn',
    ]

    if all(name in namespace for name in required_task7):
        print('-' * 62)
        print('TASK 7 - SCIKIT-LEARN COMPARISON (UNSCORED)')

        try:
            vocab = namespace['vocabulary']
            word_to_index = namespace['word_to_index']
            sk_vectorizer = namespace['sk_vectorizer']
            X_manual = np.asarray(namespace['X_manual'])
            X_sklearn = namespace['X_sklearn']

            same_vocab_size = len(vocab) == len(sk_vectorizer.vocabulary_)
            (_ok if same_vocab_size else _bad)(
                'manual and CountVectorizer vocabulary sizes match'
            )

            same_shape = X_manual.shape == X_sklearn.shape
            (_ok if same_shape else _bad)(
                'manual and CountVectorizer matrix shapes match',
                None if same_shape
                else f'manual={X_manual.shape}, sklearn={X_sklearn.shape}'
            )

            shared_words = [
                word for word in vocab
                if word in sk_vectorizer.vocabulary_
            ][:5]

            counts_match = bool(shared_words)
            for word in shared_words:
                manual_count = X_manual[0, word_to_index[word]]
                sklearn_count = X_sklearn[
                    0, sk_vectorizer.vocabulary_[word]
                ]
                counts_match = counts_match and (
                    manual_count == sklearn_count
                )

            (_ok if counts_match else _bad)(
                'sample word counts match CountVectorizer'
            )

        except Exception as e:
            _bad('Task 7 comparison diagnostic runs', repr(e))
    else:
        print(
            'Task 7 diagnostic skipped: run the Task 7 notebook cells '
            'before calling validate_lab2(globals()).'
        )

    print('=' * 62)
    return passed, total

def validate_lab3(namespace):
    print('=' * 62); print('COS 380 LAB 3 — VISIBLE SELF-CHECK'); print('=' * 62)
    passed=0; total=4
    fn=_callable(namespace,'compute_tf')
    if fn:
        try:
            tf=fn(['a','a','b']); ok=np.isclose(tf['a'],2/3) and np.isclose(tf['b'],1/3)
            (_ok if ok else _bad)('compute_tf normalized term frequencies'); passed+=int(ok)
        except Exception as e:_bad('compute_tf runs',repr(e))
    fn=_callable(namespace,'compute_idf')
    if fn:
        try:
            idf=fn([['a','b'],['a','c']]); ok=idf['a'] <= idf['b'] and idf['a'] <= idf['c']
            (_ok if ok else _bad)('compute_idf downweights widespread terms'); passed+=int(ok)
        except Exception as e:_bad('compute_idf runs',repr(e))
    fn=_callable(namespace,'cosine_similarity')
    if fn:
        try:
            ok=np.isclose(fn([1,0],[1,0]),1.0) and np.isclose(fn([1,0],[0,1]),0.0)
            (_ok if ok else _bad)('cosine_similarity basic geometry'); passed+=int(ok)
        except Exception as e:_bad('cosine_similarity runs',repr(e))
    fn=_callable(namespace,'find_similar_documents')
    if fn:
        try:
            X=np.array([[1.,0.],[.9,.1],[0.,1.]])
            r=fn(0,X,top_k=1); ok=len(r)==1 and r[0][0]==1
            (_ok if ok else _bad)('find_similar_documents ranks nearest document'); passed+=int(ok)
        except Exception as e:_bad('find_similar_documents runs',repr(e))
    print(f'Visible checks passed: {passed}/{total}')
    return passed,total


def validate_lab4(namespace):
    print('=' * 62); print('COS 380 LAB 4 — VISIBLE SELF-CHECK'); print('=' * 62)
    score=0; total=8
    fn=_callable(namespace,'build_vocabulary')
    if fn:
        try:
            vocab=fn(['red sun rises','dark moon rises'])
            ok=isinstance(vocab,dict) and vocab.get('<UNK>')==0 and len(set(vocab.values()))==len(vocab)
            (_ok if ok else _bad)('build_vocabulary reserves <UNK> at index 0'); score+=int(ok)
        except Exception as e:_bad('build_vocabulary runs',repr(e))
    fn=_callable(namespace,'tokens_to_indices')
    if fn:
        try:
            got=fn(['red','mystery','sun'],{'<UNK>':0,'red':1,'sun':2}); ok=list(got)==[1,0,2]
            (_ok if ok else _bad)('tokens_to_indices handles unseen tokens'); score+=int(ok)
        except Exception as e:_bad('tokens_to_indices runs',repr(e))
    fn=_callable(namespace,'compute_class_priors')
    if fn:
        try:
            p=fn(['poe','poe','frost']); ok=np.isclose(p['poe'],2/3) and np.isclose(p['frost'],1/3)
            (_ok if ok else _bad)('compute_class_priors basic probability check'); score+=int(ok)
        except Exception as e:_bad('compute_class_priors runs',repr(e))
    fn=_callable(namespace,'count_markov_events')
    if fn:
        try:
            pi,A=fn([[2,5,3,5]],6); ok=pi.shape==(6,) and A.shape==(6,6) and np.isclose(pi[2],1) and np.isclose(A[2,5],1) and np.isclose(A[5,3],1) and np.isclose(A[3,5],1)
            (_ok if ok else _bad)('count_markov_events counts transitions'); score+=int(ok)
        except Exception as e:_bad('count_markov_events runs',repr(e))
    fn=_callable(namespace,'build_markov_model')
    if fn:
        try:
            log_pi,log_A=fn([[1,2],[1,2]],4,alpha=1.0); pi=np.exp(log_pi); A=np.exp(log_A)
            ok1=np.isclose(pi.sum(),1.0) and np.allclose(A.sum(axis=1),1.0)
            ok2=np.all(pi>0) and np.all(A>0)
            (_ok if ok1 else _bad)('Markov distributions normalize correctly'); score+=int(ok1)
            (_ok if ok2 else _bad)('smoothing gives nonzero probabilities'); score+=int(ok2)
        except Exception as e:_bad('build_markov_model runs',repr(e))
    fn=_callable(namespace,'sequence_log_probability')
    if fn:
        try:
            lp=np.log(np.array([.25,.75])); la=np.log(np.array([[.5,.5],[.2,.8]])); got=fn([1,0,1],lp,la); exp=np.log(.75)+np.log(.2)+np.log(.5); ok=np.isclose(got,exp)
            (_ok if ok else _bad)('sequence_log_probability basic score'); score+=int(ok)
        except Exception as e:_bad('sequence_log_probability runs',repr(e))
    fn=_callable(namespace,'predict_one')
    if fn:
        try:
            models={'poe':(np.log(np.array([.9,.1])),np.log(np.array([[.1,.9],[.5,.5]]))), 'frost':(np.log(np.array([.1,.9])),np.log(np.array([[.9,.1],[.5,.5]])))}
            pred,scores=fn([0,1],models,{'poe':.5,'frost':.5}); ok=pred=='poe' and set(scores)=={'poe','frost'}
            (_ok if ok else _bad)('predict_one selects higher Bayes score'); score+=int(ok)
        except Exception as e:_bad('predict_one runs',repr(e))
    print(f'Visible checks passed: {score}/{total}')
    print('These are basic self-checks, not the complete instructor grading tests.')
    return score,total


def validate_lab5(ns):
    print("="*64)
    print("COS 380 LAB 5 — VISIBLE SELF-CHECK")
    print("="*64)

    checks = 0
    passed = 0

    def get(name):
        nonlocal checks
        checks += 1
        obj=ns.get(name)
        if not callable(obj):
            _bad(f"{name} found", "Required function is missing or not callable.")
            return None
        _ok(f"{name} found")
        return obj

    # normalization
    fn=get("normalize_counts")
    if fn:
        try:
            p=fn({"A":2,"B":3,"C":5})
            if np.isclose(sum(p.values()),1) and np.isclose(p["A"],.2) and np.isclose(p["B"],.3):
                _ok("normalize_counts() basic behavior")
                passed+=1
            else:
                _bad("normalize_counts() values", repr(p))
        except Exception as e:
            _bad("normalize_counts() runs", repr(e))

    fn=get("build_initial_distribution")
    if fn:
        try:
            p=fn([["red","sun","<END>"],["blue","sky","<END>"],["red","moon","<END>"]])
            if np.isclose(p["red"],2/3) and np.isclose(p["blue"],1/3):
                _ok("initial distribution")
                passed+=1
            else:_bad("initial distribution values",repr(p))
        except Exception as e:_bad("initial distribution runs",repr(e))

    fn=get("build_second_word_distributions")
    if fn:
        try:
            d=fn([["red","sun","<END>"],["red","moon","<END>"],["blue","sky","<END>"]])
            good=np.isclose(d["red"]["sun"],.5) and np.isclose(d["red"]["moon"],.5)
            good=good and all(np.isclose(sum(x.values()),1) for x in d.values())
            if good:_ok("second-word conditional distributions"); passed+=1
            else:_bad("second-word distributions",repr(d))
        except Exception as e:_bad("second-word distributions run",repr(e))

    fn=get("build_second_order_transitions")
    if fn:
        try:
            d=fn([
                ["the","woods","are","dark","<END>"],
                ["the","woods","are","deep","<END>"],
            ])
            good=("the","woods") in d and np.isclose(d[("the","woods")]["are"],1)
            good=good and np.isclose(d[("woods","are")]["dark"],.5)
            good=good and np.isclose(d[("woods","are")]["deep"],.5)
            if good:_ok("second-order context uses two words"); passed+=1
            else:_bad("second-order transitions",repr(d))
        except Exception as e:_bad("second-order transitions run",repr(e))

    fn=get("sample_word")
    if fn:
        try:
            rng=np.random.default_rng(123)
            probs={"A":.2,"B":.5,"C":.3}
            values=[fn(probs,rng) for _ in range(100)]
            if set(values).issubset(probs) and len(values)==100:
                _ok("sample_word() returns valid outcomes"); passed+=1
            else:_bad("sample_word() output")
        except Exception as e:_bad("sample_word() runs",repr(e))

    fn=get("generate_line")
    if fn:
        try:
            toy={
                "initial":{"hello":1.0},
                "second":{"hello":{"world":1.0}},
                "transitions":{("hello","world"):{"<END>":1.0}}
            }
            result=fn(toy,np.random.default_rng(1),max_tokens=10)
            if result=="hello world":
                _ok("generate_line() handles <END>"); passed+=1
            else:_bad("generate_line() output",f"Expected 'hello world', got {result!r}")
        except Exception as e:_bad("generate_line() runs",repr(e))

    fn=get("generate_poem")
    if fn:
        try:
            toy={
                "initial":{"hello":1.0},
                "second":{"hello":{"world":1.0}},
                "transitions":{("hello","world"):{"<END>":1.0}}
            }
            result=fn(toy,4,np.random.default_rng(1))
            if isinstance(result,list) and len(result)==4:
                _ok("generate_poem() returns requested number of lines"); passed+=1
            else:_bad("generate_poem() output",repr(result))
        except Exception as e:_bad("generate_poem() runs",repr(e))

    print("-"*64)
    print(f"Visible functional checks passed: {passed}/{checks}")
    print("These are not the complete instructor grading tests.")
    return passed, checks
