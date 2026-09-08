"""CPU regression for the experiment's actual coordinate initialization block.

Execute only its AST slice so the test cannot enter the GPU training driver.
The prior bug kept the teacher optimizer after replacing its coordinate tensor.
"""
import ast
from pathlib import Path

import torch
import pytest


def test_alternating_optimizer_owns_new_coordinates_and_clears_their_gradients():
    source = Path(__file__).resolve().parents[1] / "experiments/sandbox/exp_a_train.py"
    module = ast.parse(source.read_text())
    main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "main")

    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        )

    start = max(i for i, node in enumerate(main.body) if assigns(node, "Z"))
    end = next(i for i in range(start + 1, len(main.body)) if assigns(main.body[i], "outer"))
    init = ast.Module(body=main.body[start:end], type_ignores=[])
    teacher = torch.ones(8, 2, requires_grad=True)
    teacher_optimizer = torch.optim.Adam([teacher], lr=0.01)
    namespace = {
        "torch": torch,
        "pca2d_init": lambda: torch.arange(16, dtype=torch.float32).reshape(8, 2) + 1,
        "optZ": teacher_optimizer,
    }
    exec(compile(init, str(source), "exec"), namespace)
    coords, optimizer = namespace["Z"], namespace["optZ"]
    initial = coords.detach().clone()
    assert optimizer is not teacher_optimizer
    assert any(p is coords for group in optimizer.param_groups for p in group["params"])
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        assert coords.grad is None
        coords.square().mean().backward()
        optimizer.step()
    assert not torch.equal(coords, initial)
    assert torch.equal(teacher, torch.ones_like(teacher))


@pytest.mark.parametrize(
    "candidate_wall,candidate_cohorts,promotes",
    [(100, {"a": 0.74, "b": 0.88}, False),
     (110, {"a": 0.74, "b": 0.93}, False),
     (100, {"a": 0.74, "b": 0.93}, True)],
)
def test_quality_promotion_requires_matched_cohort_and_wall_guards(
    candidate_wall, candidate_cohorts, promotes
):
    source = Path(__file__).resolve().parents[1] / "experiments/sandbox/exp_a_train.py"
    main = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(i for i, n in enumerate(main.body)
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
                 and [t.id for t in n.targets[0].elts] == ["dc", "al"])
    end = next(i for i in range(start + 1, len(main.body))
               if isinstance(main.body[i], ast.Assign)
               and isinstance(main.body[i].targets[0], ast.Name)
               and main.body[i].targets[0].id == "out")
    results = {
        "direct": {"wall_s": 100, "recall@k15_B2000": {
            "micro": 0.80, "per_source": {"a": 0.70, "b": 0.90}}},
        "alternating": {"wall_s": candidate_wall, "recall@k15_B2000": {
            "micro": 0.83, "per_source": candidate_cohorts}},
    }
    namespace = {"results": results}
    exec(compile(ast.Module(body=main.body[start:end], type_ignores=[]), str(source), "exec"), namespace)
    assert namespace["verdict"].startswith("PROMOTE") is promotes
