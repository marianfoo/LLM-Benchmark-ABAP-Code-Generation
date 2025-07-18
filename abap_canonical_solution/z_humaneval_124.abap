CLASS z_humaneval_124 DEFINITION PUBLIC FINAL CREATE PUBLIC.
  PUBLIC SECTION.
    CLASS-METHODS: valid_date
      IMPORTING
        iv_date          TYPE string
      RETURNING
        VALUE(rv_result) TYPE abap_bool.
ENDCLASS.

CLASS z_humaneval_124 IMPLEMENTATION.
  METHOD valid_date.
    DATA: lv_date  TYPE string,
          lt_parts TYPE string_table,
          lv_month TYPE i,
          lv_day   TYPE i,
          lv_year  TYPE i.

    lv_date = condense( iv_date ).

    IF lv_date IS INITIAL.
      rv_result = abap_false.
      RETURN.
    ENDIF.

    SPLIT lv_date AT '-' INTO TABLE lt_parts.

    IF lines( lt_parts ) <> 3.
      rv_result = abap_false.
      RETURN.
    ENDIF.

    TRY.
        lv_month = lt_parts[ 1 ].
        lv_day = lt_parts[ 2 ].
        lv_year = lt_parts[ 3 ].
      CATCH cx_sy_conversion_no_number.
        rv_result = abap_false.
        RETURN.
    ENDTRY.

    IF lv_month < 1 OR lv_month > 12.
      rv_result = abap_false.
      RETURN.
    ENDIF.

    IF ( lv_month = 1 OR lv_month = 3 OR lv_month = 5 OR lv_month = 7 OR lv_month = 8 OR lv_month = 10 OR lv_month = 12 ) AND ( lv_day < 1 OR lv_day > 31 ).
      rv_result = abap_false.
      RETURN.
    ENDIF.

    IF ( lv_month = 4 OR lv_month = 6 OR lv_month = 9 OR lv_month = 11 ) AND ( lv_day < 1 OR lv_day > 30 ).
      rv_result = abap_false.
      RETURN.
    ENDIF.

    IF lv_month = 2 AND ( lv_day < 1 OR lv_day > 29 ).
      rv_result = abap_false.
      RETURN.
    ENDIF.

    rv_result = abap_true.
  ENDMETHOD.
ENDCLASS.