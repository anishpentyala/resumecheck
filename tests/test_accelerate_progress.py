"""A loader which repeatedly yields nothing cannot consume unbounded epochs."""
import subprocess
import sys
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('accelerate')
pytest.importorskip('torchdata')
from resumecheck.accelerate import check_accelerate,check_accelerate_skip_path


class Data(torch.utils.data.Dataset):
    def __len__(self):return 16
    def __getitem__(self,i):return {'x':torch.ones(1)}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
    def forward(self,x):return {'loss':self.weight.square()}


@pytest.mark.parametrize('check',[check_accelerate,check_accelerate_skip_path])
@pytest.mark.parametrize('value',[True,False,2.5,2.0,'2',None,0,-1])
def test_invalid_batch_size_precedes_factory(check,value,tmp_path):
    def forbidden():pytest.fail('invalid setup must not construct model')
    with pytest.raises((ValueError,TypeError),match='batch_size'):
        check(forbidden,Data(),4,2,batch_size=value,artifacts_dir=tmp_path/'evidence')
    assert not (tmp_path/'evidence').exists()


@pytest.mark.parametrize('check',[check_accelerate,check_accelerate_skip_path])
def test_empty_sized_dataset_has_clear_error(check):
    class Empty(Data):
        def __len__(self):return 0
    with pytest.raises(ValueError,match='nonempty sized'):
        check(Model,Empty(),4,2)


@pytest.mark.parametrize('check',[check_accelerate,check_accelerate_skip_path])
def test_integer_index_batch_size_supported(check):
    class Index:
        def __index__(self):return 2
    assert check(Model,Data(),4,2,batch_size=Index()).passed


@pytest.mark.parametrize('path',['stateful','skip'])
def test_real_adapter_repeated_empty_iterations_end_instead_of_hanging(tmp_path,path):
    # A child process bounds the buggy baseline. The sentinel is written only
    # after real Accelerate preparation, establishing the tested loop was hit.
    marker=tmp_path/'prepared'
    script=r'''
import pathlib,sys,torch
from accelerate import Accelerator
from resumecheck.accelerate import check_accelerate,check_accelerate_skip_path
class Data(torch.utils.data.Dataset):
 def __len__(self):return 16
 def __getitem__(self,i):return {'x':torch.ones(1)}
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__();self.p=torch.nn.Parameter(torch.tensor(1.))
 def forward(self,x):return {'loss':self.p.square()}
original=Accelerator.prepare
class Empty:
 def __iter__(self):return iter(())
def prepare(self,*args,**kwargs):
 model,opt,loader=original(self,*args,**kwargs)
 pathlib.Path(sys.argv[1]).write_text('prepared')
 return model,opt,Empty()
Accelerator.prepare=prepare
check=check_accelerate if sys.argv[2]=='stateful' else check_accelerate_skip_path
try:check(Model,Data(),4,2,artifacts_dir=pathlib.Path(sys.argv[1]).with_name('evidence'))
except ValueError as error:
 assert 'no progress for two consecutive' in str(error),str(error)
 assert not pathlib.Path(sys.argv[1]).with_name('evidence').exists()
 print('bounded-no-progress-error')
else:raise AssertionError('empty loader must not verify')
'''
    try:
        result=subprocess.run([sys.executable,'-c',script,str(marker),path],capture_output=True,text=True,timeout=12)
    except subprocess.TimeoutExpired:
        assert marker.exists(), 'adapter preparation not reached; timeout is not a loop witness'
        pytest.fail('prepared adapter hung on repeated empty iterations')
    assert result.returncode==0,result.stderr
    assert 'bounded-no-progress-error' in result.stdout and marker.exists()


@pytest.mark.parametrize('check',[check_accelerate,check_accelerate_skip_path])
def test_one_empty_iterator_before_training_is_allowed(monkeypatch,check):
    from accelerate import Accelerator
    prepare=Accelerator.prepare
    class EmptyFirst:
        def __init__(self,loader):self.loader=loader;self.first=True
        def __iter__(self):
            if self.first:
                self.first=False
                return iter(())
            return iter(self.loader)
        def __getattr__(self,name):return getattr(self.loader,name)
    def wrapped(self,*args,**kwargs):
        model,opt,loader=prepare(self,*args,**kwargs)
        return model,opt,EmptyFirst(loader)
    monkeypatch.setattr(Accelerator,'prepare',wrapped)
    assert check(Model,Data(),4,2).passed
