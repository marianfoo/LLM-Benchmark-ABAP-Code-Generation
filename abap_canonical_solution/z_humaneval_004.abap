CLASS z_humaneval_004 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF float_table_type,
             value TYPE f,
           END OF float_table_type,
           float_table TYPE TABLE OF float_table_type.

    CLASS-METHODS: mean_absolute_deviation
      IMPORTING
        numbers       TYPE float_table
      RETURNING
        VALUE(result) TYPE f.

  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_004 IMPLEMENTATION.
  METHOD mean_absolute_deviation.
    DATA: mean_val     TYPE f,
          sum_val      TYPE f,
          count        TYPE i,
          abs_diff_sum TYPE f.

    LOOP AT numbers INTO DATA(number).
      sum_val = sum_val + number-value.
      count = count + 1.
    ENDLOOP.

    IF count > 0.
      mean_val = sum_val / count.

      LOOP AT numbers INTO number.
        abs_diff_sum = abs_diff_sum + abs( number-value - mean_val ).
      ENDLOOP.

      result = abs_diff_sum / count.
    ENDIF.
  ENDMETHOD.
ENDCLASS.