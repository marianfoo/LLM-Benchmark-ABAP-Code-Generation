CLASS z_humaneval_026 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.

    CLASS-METHODS remove_duplicates
      IMPORTING
        VALUE(numbers) TYPE int4_table
      RETURNING
        VALUE(result)  TYPE int4_table.

  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_026 IMPLEMENTATION.

  METHOD remove_duplicates.
    DATA: lt_numbers TYPE int4_table,
          lt_unique  TYPE int4_table,
          lv_number  TYPE i.

    " Sort the input table for efficient processing
    lt_numbers = numbers.
    SORT lt_numbers.

    " Loop over sorted numbers and count occurrences
    LOOP AT lt_numbers INTO lv_number.
      DATA(count) = 0.
      LOOP AT lt_numbers INTO DATA(temp) WHERE table_line = lv_number.
        count = count + 1.
      ENDLOOP.
      " If count is exactly 1, append to result
      IF count = 1.
        " To avoid duplicates in result (if numbers appeared multiple times)
        READ TABLE lt_unique WITH KEY table_line = lv_number TRANSPORTING NO FIELDS.
        IF sy-subrc <> 0.
          APPEND lv_number TO lt_unique.
        ENDIF.
      ENDIF.
    ENDLOOP.

    result = lt_unique.
  ENDMETHOD.

ENDCLASS.
