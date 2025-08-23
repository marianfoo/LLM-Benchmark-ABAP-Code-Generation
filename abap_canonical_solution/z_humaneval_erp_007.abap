CLASS z_humaneval_erp_007 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .
  PUBLIC SECTION.

    CLASS-METHODS: calculate_delivery_date
      IMPORTING
        order_date           TYPE string
        mat_id               TYPE string
      RETURNING
        VALUE(rv_delivery_date) TYPE dats.

  PRIVATE SECTION.
    CLASS-METHODS: get_lead_time
      IMPORTING
        iv_mat_id           TYPE string
      RETURNING
        VALUE(rv_lead_time) TYPE i,
      is_working_day
        IMPORTING
          iv_date                  TYPE dats
        RETURNING
          VALUE(rv_is_working_day) TYPE abap_bool,
      is_holiday
        IMPORTING
          iv_date              TYPE dats
        RETURNING
          VALUE(rv_is_holiday) TYPE abap_bool,
      add_working_days
        IMPORTING
          iv_start_date      TYPE dats
          iv_days_to_add     TYPE i
        RETURNING
          VALUE(rv_end_date) TYPE dats.
ENDCLASS.

CLASS z_humaneval_erp_007 IMPLEMENTATION.
  METHOD calculate_delivery_date.
    DATA: lv_order_date TYPE dats,
          lv_lead_time  TYPE i.
    " Convert string date to DATS format
    lv_order_date = order_date.
    " Get lead time for material
    lv_lead_time = get_lead_time( mat_id ).
    " If no lead time, return order date
    IF lv_lead_time = 0.
      rv_delivery_date = lv_order_date.
      RETURN.
    ENDIF.
    " Add working days to order date
    rv_delivery_date = add_working_days( iv_start_date = lv_order_date
                                        iv_days_to_add = lv_lead_time ).
  ENDMETHOD.

  METHOD get_lead_time.
    DATA: lt_material TYPE TABLE OF zmaterial.
    " Select material data from database table
    SELECT mat_id, lead_time_days
      FROM zmaterial
      INTO TABLE @lt_material
      WHERE mat_id = @iv_mat_id.
    " If material found, return lead time, otherwise default to 0
    READ TABLE lt_material INTO DATA(ls_material) INDEX 1.
    IF sy-subrc = 0.
      rv_lead_time = ls_material-lead_time_days.
    ELSE.
      rv_lead_time = 0.
    ENDIF.
  ENDMETHOD.

  METHOD is_working_day.
    DATA: lv_weekday TYPE c LENGTH 1.
    " Check if it's a holiday first
    IF is_holiday( iv_date ) = abap_true.
      rv_is_working_day = abap_false.
      RETURN.
    ENDIF.
    " Get weekday using DATE_COMPUTE_DAY function module
    " Returns: 1=Monday, 2=Tuesday, ..., 6=Saturday, 7=Sunday
    CALL FUNCTION 'DATE_COMPUTE_DAY'
      EXPORTING
        date = iv_date
      IMPORTING
        day  = lv_weekday.
    " Working days are Monday (1) to Friday (5)
    IF lv_weekday >= '1' AND lv_weekday <= '5'.
      rv_is_working_day = abap_true.
    ELSE.
      rv_is_working_day = abap_false.
    ENDIF.
  ENDMETHOD.

  METHOD is_holiday.
    DATA: lt_holidays TYPE TABLE OF zholiday.
    " Select holidays from database table
    SELECT holiday_date, name
      FROM zholiday
      INTO TABLE @lt_holidays
      WHERE holiday_date = @iv_date.
    " If holiday found, return true
    IF lines( lt_holidays ) > 0.
      rv_is_holiday = abap_true.
    ELSE.
      rv_is_holiday = abap_false.
    ENDIF.
  ENDMETHOD.

  METHOD add_working_days.
    DATA: lv_current_date TYPE dats,
          lv_days_added   TYPE i.
    lv_current_date = iv_start_date.
    lv_days_added = 0.
    " Loop until we've added the required number of working days
    WHILE lv_days_added < iv_days_to_add.
      " Move to next day
      lv_current_date = lv_current_date + 1.
      " Check if it's a working day
      IF is_working_day( lv_current_date ) = abap_true.
        lv_days_added = lv_days_added + 1.
      ENDIF.
    ENDWHILE.
    rv_end_date = lv_current_date.
  ENDMETHOD.
ENDCLASS.