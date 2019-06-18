


for(i in list.files("man"))
{
  p = paste0("man/", i)
  tmp = readChar(p, file.info(p)$size)
  tmp = gsub("SJMC::", "SimJoint::", tmp)
  writeChar(tmp, p, eos = NULL)
}





















































