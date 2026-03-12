#!/bin/bash
# This script loads all of the NIBRS data stored in ./<STATE>/<STATE>-<YEAR>
# to a postgresql database called scotfreedb. 


<<'###'
select vic.victim_id, off.offense_id, circ.victim_id from nibrs_offense as off
inner join nibrs_victim as vic on off.incident_id = vic.incident_id 
inner join nibrs_victim_circumstances as circ on circ.victim_id = vic.victim_id
 limit 20;




# 1) Make sure that all the data folders have the needed load.sql file
goodtogo=true
for state in ??; do
    for datadir in $state/??-????; do
        if [ -f $datadir/postgres_load.sql ]; then
            #echo $datadir/postgres_load.sql
            true
        else
            echo "warning: $datadir/postgres_load.sql doesn't exist"
            goodtogo=false
        fi
    done
done


# 2) Inform user that the data may be incomplete. Ask to go for it anyway
if [ $goodtogo == false ]; then
    read -p "one or more data folders are missing postgres_load.sql files. continue? (y/n) " response
    if [ "$response" != "y" ]; then
        echo "data loading aborted"
        exit 0
    else
        true
    fi
fi

###

# 1) Create scotfreedb with postgres_master_setup.sql
psql <INSERT POSTGRES SERVER URI HERE> \
-c "DROP DATABASE IF EXISTS scotfreedb;" -c "CREATE DATABASE scotfreedb;" -c "VACUUM FULL" # idempotency über alles
psql <INSERT POSTGRES SERVER URI HERE> < postgres_master_setup.sql

#SET default_transaction_read_only = OFF;



# 2) Edit the loading script, clean out non-homicides from the offense csv, load the data (and delete irrelevant data) in one loop
for state in NH ME MA RI VT CT; do # for state in ??; do
    for datadir in $state/??-??2[234]; do # for datadir in $state/??-????; do
        if [ -d $datadir ]; then
            # 2a) Remove data relating to non-murderous offenses from offense and arrestee CSVs
            
            # offense CSV purge
            offense_csv=$(find $datadir -iname 'nibrs_offense.csv')
            if [ -f "${offense_csv}.full" ]; then #the script has been run before; don't run sed -i
                true
            elif [ -f $offense_csv ]; then
                sed -i'.full' -E '/09A|data_year/I!d' $offense_csv # 09A corresponds to murders and nonnegligent manslaughter as per nibrs_offense_type.csv
                #sed -i'.full' -E '/^[^,]*,[^,]*,[^,]*,32,|offense_type_id/I!d' $offense_csv # code 32 corresponds to murders and nonnegligent manslaughter as per nibrs_offense_type.csv
            fi

            # arrestee CSV purge
            arrestee_csv=$(find $datadir -iname 'nibrs_arrestee.csv')
            if [ -f "${arrestee_csv}.full" ]; then #the script has been run before; don't run sed -i
                true
            elif [ -f $arrestee_csv ]; then
                sed -i'.full' -E '/09A|data_year/I!d' $arrestee_csv # 09A corresponds to murders and nonnegligent manslaughter as per nibrs_offense_type.csv
            fi

        
            # 2b) truncate the reference tables (identical between data batches) and load the data with postgres_master_load.sql
            psql <INSERT POSTGRES SERVER URI HERE> -c "TRUNCATE ref_race, ref_state, 
            nibrs_using_list, nibrs_ethnicity, nibrs_cleared_except, nibrs_criminal_act_type, nibrs_offense_type, nibrs_activity_type, nibrs_arrest_type, nibrs_suspected_drug_type, 
            nibrs_prop_desc_type, nibrs_prop_loss_type, nibrs_location_type, nibrs_injury, nibrs_relationship, nibrs_justifiable_force, nibrs_age, nibrs_assignment_type, nibrs_bias_list, 
            nibrs_circumstances, nibrs_criminal_act_type, nibrs_drug_measure_type, nibrs_weapon_type, nibrs_victim_type;"

            # 2c) copy master load script to the data directory, use it to load data in postgres server, then delete master load script
            cp postgres_master_load.sql $datadir/postgres_master_load.sql
            (cd $datadir && psql <INSERT POSTGRES SERVER URI HERE> < postgres_master_load.sql)
            rm $datadir/postgres_master_load.sql # remove it after using

            # 2d) remove data pertaining to nonhomicides (to preserve space)
            psql <INSERT POSTGRES SERVER URI HERE> -c "
            DELETE FROM nibrs_arrestee WHERE offense_code != '09A';
        

            DELETE FROM nibrs_victim WHERE incident_id NOT IN(
            SELECT incident_id FROM nibrs_offense);
        
        
            DELETE FROM nibrs_weapon where offense_id NOT IN(
            SELECT offense_id FROM nibrs_offense);
        

            DELETE FROM nibrs_victim_offense WHERE offense_id NOT IN(
            SELECT offense_id from nibrs_offense);
        

            DELETE FROM nibrs_victim_offender_rel where victim_id NOT IN(
            SELECT victim_id from nibrs_victim);
        

            DELETE FROM nibrs_victim_injury where victim_id NOT IN(
            SELECT victim_id from nibrs_victim);
        

            DELETE FROM nibrs_victim_circumstances where victim_id NOT IN(
            SELECT victim_id from nibrs_victim);
        

            --DELETE FROM nibrs_suspected_drug where

            DELETE FROM nibrs_suspect_using where offense_id NOT IN(
            SELECT offense_id from nibrs_offense);

            --DELETE FROM nibrs_property WHERE incident_id NOT IN(
            "

            psql <INSERT POSTGRES SERVER URI HERE> -c "VACUUM;"
        fi


    done
done

# 3) Load these two LUT-style CSVs now-- format is not consistent across states/years 
psql <INSERT POSTGRES SERVER URI HERE> \
-c "\COPY NIBRS_OFFENSE_TYPE FROM 'model_NIBRS_OFFENSE_TYPE.csv' DELIMITER ',' HEADER CSV;" \
-c "\COPY NIBRS_PROP_DESC_TYPE FROM 'model_NIBRS_PROP_DESC_TYPE.csv' DELIMITER ',' HEADER CSV;" \
-c "\COPY NIBRS_RELATIONSHIP FROM 'model_NIBRS_RELATIONSHIP.csv' DELIMITER ',' HEADER CSV;"







#psql -U your_username -d your_database_name -f postgres_load.sql
#psql aiven-url < postgresfile.sql
