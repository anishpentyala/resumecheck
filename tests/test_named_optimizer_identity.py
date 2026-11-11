"""Live ownership matters even when serialized state_dict bytes agree."""
import copy
import pytest

torch = pytest.importorskip('torch')
from resumecheck.optimizer_identity import fingerprint_named_optimizer
from resumecheck.recorder import fingerprint_tensor, StepRecord, Trajectory
from resumecheck.compare import compare_trajectories


class Pair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor([10.0], dtype=torch.float64))
        self.right = torch.nn.Parameter(torch.tensor([20.0], dtype=torch.float64))


def optimizer(model, kind=torch.optim.SGD, reverse=False, split=False):
    params=[model.left,model.right]
    if reverse: params.reverse()
    if split:params=[{'params':[p]} for p in params]
    kwargs={'lr':0.125}
    if kind is torch.optim.SGD:kwargs['momentum']=0.9
    return kind(params,**kwargs)


def update(model, opt):
    model.left.grad=torch.tensor([2.0],dtype=torch.float64)
    model.right.grad=torch.tensor([8.0],dtype=torch.float64)
    opt.step(); opt.zero_grad()


@pytest.mark.parametrize('kind',[torch.optim.SGD,torch.optim.Adam,torch.optim.AdamW])
@pytest.mark.parametrize('split',[False,True])
def test_same_raw_state_can_have_wrong_live_owners(kind,split):
    source=Pair(); original=optimizer(source,kind,split=split);update(source,original)
    saved=copy.deepcopy(original.state_dict())
    good=Pair();good.load_state_dict(source.state_dict())
    bad=Pair();bad.load_state_dict(source.state_dict())
    correct=optimizer(good,kind,split=split);wrong=optimizer(bad,kind,True,split)
    correct.load_state_dict(copy.deepcopy(saved));wrong.load_state_dict(copy.deepcopy(saved))
    assert fingerprint_tensor(correct.state_dict()) == fingerprint_tensor(wrong.state_dict())
    assert fingerprint_tensor(good.state_dict()) == fingerprint_tensor(bad.state_dict())
    assert fingerprint_named_optimizer(good,correct) != fingerprint_named_optimizer(bad,wrong)
    a=Trajectory('good',[StepRecord(1,loss=1.0,optimizer_fingerprint=fingerprint_named_optimizer(good,correct))])
    b=Trajectory('bad',[StepRecord(1,loss=1.0,optimizer_fingerprint=fingerprint_named_optimizer(bad,wrong))])
    report=compare_trajectories(a,b)
    assert not report.passed and report.first_divergence.field=='optimizer_state'


def test_named_metadata_cannot_override_live_object_binding():
    source=Pair();opt=optimizer(source);update(source,opt)
    state=copy.deepcopy(opt.state_dict());state['param_groups'][0]['param_names']=['left','right']
    a,b=Pair(),Pair();good,bad=optimizer(a),optimizer(b,reverse=True)
    good.load_state_dict(copy.deepcopy(state));bad.load_state_dict(copy.deepcopy(state))
    assert good.param_groups[0]['param_names']==bad.param_groups[0]['param_names']
    assert fingerprint_named_optimizer(a,good)!=fingerprint_named_optimizer(b,bad)


@pytest.mark.parametrize('kind',[torch.optim.SGD,torch.optim.Adam,torch.optim.AdamW])
def test_fresh_equal_instances_and_lazy_empty_state(kind):
    a,b=Pair(),Pair();oa,ob=optimizer(a,kind),optimizer(b,kind)
    assert fingerprint_named_optimizer(a,oa)==fingerprint_named_optimizer(b,ob)


def test_model_registration_order_does_not_control_binding():
    a,b=Pair(),Pair()
    b._parameters=dict(reversed(list(b._parameters.items())))
    assert fingerprint_named_optimizer(a,optimizer(a))==fingerprint_named_optimizer(b,optimizer(b))


def test_partial_model_optimizer_supported():
    a,b=Pair(),Pair()
    oa=torch.optim.SGD([a.left],lr=.1);ob=torch.optim.SGD([b.left],lr=.1)
    assert fingerprint_named_optimizer(a,oa)==fingerprint_named_optimizer(b,ob)


@pytest.mark.parametrize('change',['alias','foreign','duplicate','empty','hook_pre','hook_post',
                                    'subclass','sparse','noncpu'])
def test_unqualified_identity_fails_closed(change):
    model=Pair();opt=optimizer(model)
    if change=='alias':model.alias=model.left
    if change=='foreign':opt.param_groups[0]['params'][0]=torch.nn.Parameter(torch.ones(1))
    if change=='duplicate':opt.param_groups[0]['params'][1]=model.left
    if change=='empty':opt.param_groups[0]['params']=[]
    if change=='hook_pre':opt.register_state_dict_pre_hook(lambda o:None)
    if change=='hook_post':opt.register_state_dict_post_hook(lambda o,s:s)
    if change=='subclass':
        class Custom(torch.optim.SGD):pass
        opt=Custom(model.parameters(),lr=.1)
    if change=='sparse':
        model.left=torch.nn.Parameter(torch.sparse_coo_tensor([[0]],[1.],(1,),check_invariants=True))
        opt=optimizer(model)
    if change=='noncpu':
        model.left=torch.nn.Parameter(torch.empty(1,device='meta'));opt=optimizer(model)
    with pytest.raises((TypeError,ValueError)):
        fingerprint_named_optimizer(model,opt)


def test_call_does_not_mutate_model_or_optimizer():
    model=Pair();opt=optimizer(model);update(model,opt)
    before=(fingerprint_tensor(model.state_dict()),fingerprint_tensor(opt.state_dict()))
    fingerprint_named_optimizer(model,opt)
    assert before==(fingerprint_tensor(model.state_dict()),fingerprint_tensor(opt.state_dict()))


@pytest.mark.parametrize('change',['lr','momentum','requires_grad','shape','dtype','buffer'])
def test_ordinary_observed_changes_still_differ(change):
    a,b=Pair(),Pair();oa,ob=optimizer(a),optimizer(b);update(a,oa);update(b,ob)
    if change=='lr':ob.param_groups[0]['lr']=.01
    if change=='momentum':ob.param_groups[0]['momentum']=.8
    if change=='requires_grad':b.left.requires_grad_(False)
    if change=='shape':b.left.data=b.left.data.reshape(1,1)
    if change=='dtype':b.left.data=b.left.data.float()
    if change=='buffer':ob.state[b.left]['momentum_buffer'].add_(1)
    assert fingerprint_named_optimizer(a,oa)!=fingerprint_named_optimizer(b,ob)


def test_invalid_model_type():
    with pytest.raises(TypeError):fingerprint_named_optimizer(None,None)


def test_sgd_continuation_and_schema4_replay(tmp_path):
    from resumecheck.artifacts import save_comparison_artifacts, replay_comparison_artifacts
    from dataclasses import asdict
    source=Pair();source_opt=optimizer(source);update(source,source_opt)
    saved=copy.deepcopy(source_opt.state_dict())
    good,bad=Pair(),Pair()
    good.load_state_dict(source.state_dict());bad.load_state_dict(source.state_dict())
    correct,wrong=optimizer(good),optimizer(bad,reverse=True)
    correct.load_state_dict(copy.deepcopy(saved));wrong.load_state_dict(copy.deepcopy(saved))
    for model,opt in [(good,correct),(bad,wrong)]:
        model.left.grad=torch.zeros_like(model.left)
        model.right.grad=torch.zeros_like(model.right)
        opt.step();opt.zero_grad()
    torch.testing.assert_close(good.left,torch.tensor([9.525],dtype=torch.float64))
    torch.testing.assert_close(bad.left,torch.tensor([8.85],dtype=torch.float64))
    # A scalar-only observation and positional optimizer dict still agree.
    assert fingerprint_tensor(correct.state_dict())==fingerprint_tensor(wrong.state_dict())
    raw_a=Trajectory('good',[StepRecord(2,loss=0.,optimizer_fingerprint=fingerprint_tensor(correct.state_dict()))])
    raw_b=Trajectory('bad',[StepRecord(2,loss=0.,optimizer_fingerprint=fingerprint_tensor(wrong.state_dict()))])
    assert compare_trajectories(raw_a,raw_b).passed
    named_a=Trajectory('good',[StepRecord(2,loss=0.,optimizer_fingerprint=fingerprint_named_optimizer(good,correct))])
    named_b=Trajectory('bad',[StepRecord(2,loss=0.,optimizer_fingerprint=fingerprint_named_optimizer(bad,wrong))])
    report=compare_trajectories(named_a,named_b)
    assert not report.passed and report.first_divergence.field=='optimizer_state'
    save_comparison_artifacts(tmp_path/'evidence',named_a,named_b,report,rtol=1e-5,atol=1e-7)
    assert asdict(replay_comparison_artifacts(tmp_path/'evidence'))==asdict(report)


@pytest.mark.parametrize('registry', ['_optimizer_state_dict_pre_hooks',
                                      '_optimizer_state_dict_post_hooks'])
@pytest.mark.parametrize('value', [None, [], False])
def test_unknown_hook_registry_layout_refused(registry, value):
    model=Pair();opt=optimizer(model)
    setattr(opt,registry,value)
    with pytest.raises(ValueError,match='registry layout'):
        fingerprint_named_optimizer(model,opt)
