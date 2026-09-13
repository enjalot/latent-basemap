"""CPU-only complete raw-history layout check before035 uses the GPU."""
def expected_ranges(low=19344847,high=103816750,chunk=32768):
 assert 0<low<high and chunk>0
 return [(lo,min(lo+chunk,end)) for base,end in [(0,low),(low,high)] for lo in range(base,end,chunk)]
def validate_raw_history(history,low=19344847,high=103816750,chunk=32768):
 expected=expected_ranges(low,high,chunk)
 assert len(history)==len(expected),'raw history incomplete or duplicated'
 for h,pair in zip(history,expected):
  assert (h['lo'],h['hi'])==pair,('raw history chunk layout mismatch',pair)
  digest=h['raw_sha'];assert isinstance(digest,str) and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'malformed raw content digest'
 return {'PASS':True,'chunks':len(expected),'rows':high,'segment_boundary':low,'chunk':chunk,'boundary_chunks':[list(v) for v in expected if v[1]==low or v[0]==low],'scope':'Complete ordered chunk boundaries and digest syntax. Actual content identity remains checked on every streamed chunk.'}
