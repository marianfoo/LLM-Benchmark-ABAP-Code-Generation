CLASS z_humaneval_113 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: string_table TYPE STANDARD TABLE OF string WITH DEFAULT KEY.
    CLASS-METHODS: odd_count
      IMPORTING
        lst           TYPE string_table
      RETURNING
        VALUE(result) TYPE string_table.

  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_113 IMPLEMENTATION.

  METHOD odd_count.
    DATA: lv_count      TYPE i,
          lv_digit      TYPE c LENGTH 1,
          lv_digit_num  TYPE i,
          lv_count_str  TYPE string,
          lv_result_str TYPE string,
          lv_offset     TYPE i.

    LOOP AT lst INTO DATA(lv_string).
      CLEAR: lv_count.

      DO strlen( lv_string ) TIMES.
        lv_offset = sy-index - 1.
        lv_digit = lv_string+lv_offset(1).
        lv_digit_num = lv_digit.
        IF lv_digit_num MOD 2 = 1.
          lv_count = lv_count + 1.
        ENDIF.
      ENDDO.

      lv_count_str = |{ lv_count }|.
      lv_result_str = |the number of odd elements { lv_count_str }n the str{ lv_count_str }ng { lv_count_str } of the { lv_count_str }nput.|.
      APPEND lv_result_str TO result.
    ENDLOOP.

  ENDMETHOD.

ENDCLASS.