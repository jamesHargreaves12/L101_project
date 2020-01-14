from baseline.run_system import get_top_k_titles_mlp
from baseline.utils import extract_titles_from_evidence
from baseline.allennlp_test import get_noun_phrases
from baseline.mediawiki_api import get_doc_meta_for_query
from baseline.results import get_possible_docs, filter_on_title
from neon_test import MyModel

mlp_model = MyModel("model", batch_size=1)
mlp_model.load_from_path("models/test/mlp_full_epoch_300.mdl")


claim = 'Youtube is not a website.'
possible_docs = get_possible_docs(claim)
nps = get_noun_phrases(claim)
np = "Youtube"

print("Docs =", [x["title"] for x in possible_docs])
print("Nps =", nps)
metas = get_doc_meta_for_query(np)
print("Meta =", [x["title"] for x in metas])
filtered = filter_on_title(metas, claim)
print("filtered =", [x["title"] for x in filtered])

print("MLP:", get_top_k_titles_mlp(possible_docs,claim,mlp_model))


evid = [[[181689, 193927, "Arizona", 0]], [[181689, 193928, "Arizona", 1]], [[181689, 193929, "Arizona", 2]], [[181689, 193930, "Arizona", 3], [181689, 193930, "Phoenix,_Arizona", 0]], [[181689, 193932, "Arizona", 9]], [[181710, 193954, "Arizona", 0]], [[181710, 193955, "Arizona", 1], [181710, 193955, "Mountain_States", 6]], [[181710, 193956, "Arizona", 2]], [[181710, 193957, "Arizona", 3], [181710, 193957, "Phoenix,_Arizona", 0]], [[181710, 193958, "Arizona", 4]], [[181710, 193959, "Arizona", 9]], [[181710, 193960, "Arizona", 22], [181710, 193960, "State_songs_of_Arizona", 0]]]
extract_titles_from_evidence(evid)
# = {'Arizona', 'Phoenix,_Arizona', 'Mountain_States', 'State_songs_of_Arizona'}
# Arizona and Pheonix, Arizona in ^
# Arizona in MLP
claims = ['The NAACP Image Award for Outstanding Supporting Actor in a Drama Series was last given in 1995.', 'The album One of the Boys contains I Kissed a Girl.', 'The Battle of France was fought during the Second World War.', 'Arizona is not a state.', 'In the End was released through Interscope Records.', 'Arizona is not a part of the United States.', "Recovery is Eminem's first album.", 'The first inauguration of Bill Clinton made him the 22nd Super Bowl Most Valuable Player.', 'The NAACP Image Award for Outstanding Supporting Actor in a Drama Series has been received by Omar Epps.', 'Edmund H. North was a Gemini.', 'The CONCACAF Champions League is organized for dead bodies.', 'Bonn is the birthplace of Ludwig van Beethoven.', 'Edmund H. North was born June 12, 1911.', 'The CONCACAF Champions League is organized for dead bodies.', 'Down with Love is a romantic comedy movie from 2003.', 'In the End was released through Warner Bros entertainment company.', "Diana, Princess of Wales's funeral took place on September 6th, 1997.", "Diana, Princess of Wales's funeral did not start on September 6th, 1997.", 'The University of Illinois at Chicago is located in Buffalo, New York.', 'Arizona is not in the United States.', 'The series finale of Make It or Break It ended on the 5th month of the calendar year.', 'The CONCACAF Champions League is a dead body.', 'The King and I is based on a novel by an American writer born in 1903.', 'Liam Neeson has been nominated for a British Academy of Film and Television Arts Award for Best Actor.', 'The State of Palestine lays claim to a region on the eastern coast of the Mediterranean Sea.', 'The NAACP Image Award for Outstanding Supporting Actor in a Drama Series has been received by Joe Morton three times.']