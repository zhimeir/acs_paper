library(ggplot2)
library(ggpubr)
library(tidyverse)
library(showtext)
library(readr)
library(tidyverse)
showtext_auto()
# font_add("Times New Roman","/System/Library/Fonts/Supplemental/Times New Roman.ttf")
# Palette <- c("ACS  "="#1c84c6",
#              "CS"="#ed5565",
#              "ACS-aug  "="#77AD78", 
#              "WCP  "="#f8ac59", 
#              "D-WRCP  "="#A39171")
Palette <- c("ACS-aug  "="#1c84c6",
             "ACS  " = "#77AD78",
             "CS"="#ed5565")
theme_font = theme(
      legend.text = element_text(size = 13, family = "Times New Roman"),
      legend.title = element_text(size = 13, family = "Times New Roman" ),
      axis.title.x = element_text(size = 14, family = "Times New Roman" ),
      axis.text.x = element_text(size = 13, family = "Times New Roman" ),
      axis.title.y = element_text(size = 13, family = "Times New Roman"),
      axis.text.y = element_text(size = 13, family = "Times New Roman"),
      strip.text = element_text(size = 13, family = "Times New Roman"),
      #panel.title = element_text(size = 12, family = "Times New Roman"),
      legend.background = element_blank())

regressor_list = c("rf","logistic","xgbrf")
seed_batchsize = 10
num_batch = 2
level = 0.1

# this.dir = dirname(rstudioapi::getSourceEditorContext()$path)
# setwd(this.dir)
getwd()

# for(model in c("trained","opt-13b","llama-2-13b-chat-hf")){
for(model in c("opt-13b","llama-2-13b-chat-hf")){
  if(model=="trained"){
    data_seq = "cxr"
  }
  else{
    data_seq = c("triviaqa", "coqa")
  }
  for(data in data_seq){
    for(split_id in c(1,3)){
      for(split_tune_id in c(2)){
        if(split_id + split_tune_id < 10){
          
          all_res = c()
          
          for(size in c(100,500,1000)){
            for(seed_grp in 1:num_batch - 1){
              for(regressor in regressor_list){
                for(q_id in 5*(1:9)){
                  filename = sprintf("./acs_aug_results/model%s_data%s/model%s_data%s_N%d_q%d_split%d_splittune%d_method%s.csv", model, data, model, data, size, q_id, split_id, split_tune_id, regressor)
                  new_res = read_csv(filename)[,-1] 
                  new_res[,5] = rep(model, nrow(new_res))
                  new_res = new_res %>% mutate(level = q_id / 100, regressor = factor(regressor, regressor_list, c("RF", "logistic", "XGBRF")), model = model, size = size)
                  all_res = rbind(all_res, new_res)
                }
              }
            }
          }
          
          
          head(all_res)
          sum_res = all_res %>% group_by(level, method, regressor, model, size) %>%
            mutate(method = factor(method, c("acsaug", "adacs", "CS"), c("ACS-aug  ", "ACS  ", "CS"))) %>%
            summarise(FDR = mean(fdp), Power = mean(power), fdr_sd = sd(fdp)/sqrt(seed_batchsize * num_batch), power_sd = sd(power)/sqrt(seed_batchsize * num_batch)) %>%
            filter(level <=2)
          
          
          
          pp1 = sum_res %>%
            ggplot( aes(x = level, y = Power,  col = method, fill = method)) +
            geom_bar(stat = "identity", position = position_dodge(0.035), alpha = 0.7, width = 0.03) +
            theme_bw() +
            # geom_point() +
            # geom_line() +
            # theme_bw() +
            # geom_smooth(aes(ymin = Power - 2 * power_sd,
            #                 ymax = Power + 2 * power_sd,
            #                 group = method,
            #                 fill = method),
            #             stat = 'identity', alpha = 0.2) +
            # facet_grid(model ~ regressor)+
            facet_grid(size ~ regressor)+
            scale_colour_manual(values=Palette) +
            scale_fill_manual(values=Palette) +
            theme_font +
            theme(legend.title = element_blank())
          print(pp1)
          
          pp2 = sum_res %>%
            ggplot( aes(x = level, y = FDR,  col = method, fill = method)) +
            geom_bar(stat = "identity", position = position_dodge(0.035), alpha = 0.7, width = 0.03) +
            theme_bw() +
            # geom_point() +
            # geom_line() +
            # theme_bw() +
            # ylim(0,1) +
            geom_abline(slope=1, intercept=0, linetype='dashed') +
            # # geom_hline(yintercept = level, col = "red", linetype = "dashed") +
            # geom_smooth(aes(ymin = FDR - 2 * fdr_sd,
            #                 ymax = FDR + 2 * fdr_sd,
            #                 group = method,
            #                 fill = method),
            #             stat = 'identity', alpha = 0.2) +
            # facet_grid(model ~ regressor)+
            facet_grid(size ~ regressor)+
            scale_colour_manual(values=Palette) +
            scale_fill_manual(values=Palette) + 
            theme_font
          
          
          both = ggarrange(pp1, NULL, pp2, widths = c(1,0.1,1), common.legend = TRUE, nrow = 1)
          main_dir = sprintf("./figs/acs_aug/model%s_data%s", model, data)
          plot_dir = sprintf("./figs/acs_aug/aug_model%s_data%s_split%d_splittune%d.pdf", model, data, split_id, split_tune_id)
          # if(!dir.exists(main_dir)){
          #   dir.create(main_dir)
          # }
          ggsave(plot_dir,
                 both, width = 16.5,height = 6)
          
        }
      }
    }
  } 
}
  
