#' @export

betabart.hurdle.validation <- function(y.predict, y){
  num.iter <- dim(y.predict)[1]
  y.predict <- y.predict %>% t %>% as.data.frame()

  colnames(y.predict) <- paste0("iter[",1:num.iter,"]")

  fig <- y.predict %>%
    gather(key= chain,value = value,`iter[1]`:`iter[2500]`) %>%
    ggplot(aes(x=value,color=chain)) +
    scale_color_manual(values = c(rep("lightblue",num.iter),"black") )+
    #  scale_color_manual(values = c(rep("lightblue",582),"black") )+
    geom_density(alpha=0.5, position = "identity")  +
    guides(color="none")+
    geom_density(aes(s),data=data.frame(s=y),color="black")+
    labs(title=paste0("Treatment-","mt-cpp"))

  fig
}
