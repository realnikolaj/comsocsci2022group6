import re, nltk
from pdfminer.high_level import extract_text

htmltext = 	"<div class=\"bjelke\">Den fulde tekst</div><p class=\"Titel2\">Bekendtgørelse om delegation af social- og indenrigsministerens beføjelser efter akt nr. 179 af 20. maj 2020 om kompensation for løn og faste omkostninger til foreninger, selvejende institutioner og fonde m.v. med primært offentlig finansiering, der er i økonomisk krise som følge af Corona-virussygdom 2019 (COVID-19)</p>\r\n<p class=\"Indledning2\">I medfør af akt nr. 179 af 20. maj 2020 fastsættes:</p>\r\n<p class=\"Paragraf\" id=\"idaf8a5209-e3a4-464e-9453-db05d75901d2\">\r\n   <span id=\"P1\">\r\n   </span>\r\n   <span class=\"ParagrafNr\" id=\"Par1\">§ 1.</span> Social- og indenrigsministerens beføjelser efter akt nr. 179 af 20. maj 2020 til at fastsætte nærmere regler om lønkompensation og kompensation for faste omkostninger relateret til COVID-19 til foreninger, selvejende institutioner og fonde m.v., hvor offentlig tilskud til drift udgør halvdelen eller mere af institutionens ordinære driftsudgifter, herunder at fastsætte bestemmelser om kriterierne for at opnå kompensation, ansøgningsform, tidsfrister, betingelser for kompensation, dokumentation, modtagerkreds, udbetaling af kompensation, efterregulering og tilbagebetaling af kompensation, rentebetaling, regnskab, revision, rapportering, fremlæggelse af revisorerklæring og efterfølgende kontrol m.v., delegeres fra social- og indenrigsministeren til Socialstyrelsen.</p>\r\n<p class=\"Paragraf\" id=\"ide00824f8-17a8-4115-9699-ad8d46020bbd\">\r\n   <span id=\"P2\">\r\n   </span>\r\n   <span class=\"ParagrafNr\" id=\"Par2\">§ 2.</span> Bekendtgørelsen træder i kraft den 30. juni 2020.</p><div class=\"Givet\" id=\"Givet\"><p class=\"Givet\" align=\"center\">Social- og Indenrigsministeriet, den 25. juni 2020</p><p class=\"Sign1\" align=\"center\">Astrid Krag</p><p class=\"Sign2\" align=\"right\">/ Ditte Rex</p></div>"
pdftext = extract_text('../data/B20200098605.pdf')

pdfmatch = re.findall(r"akt nr\..*?\d{4}",pdftext)
print('pdfmatch:',pdfmatch)
htmlmatch = re.findall(r"akt nr\..*?\d{4}",htmltext)
print('htmlmatch:',htmlmatch)

# Toknize and remove punctuation
pdftokens = nltk.tokenize.word_tokenize(re.sub(r'[^\w\s]|\d','',pdftext.lower()))
# Remove stopwords
stopwords = nltk.corpus.stopwords.words('danish')
pdftokens = [w for w in pdftokens if w not in stopwords]

print(pdftokens)