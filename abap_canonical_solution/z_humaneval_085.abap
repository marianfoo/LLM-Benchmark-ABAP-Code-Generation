CLASS z_humaneval_085 DEFINITION PUBLIC.
  PUBLIC SECTION.
    CLASS-METHODS:
      add
        IMPORTING
          VALUE(lst)    TYPE int4_table
        RETURNING
          VALUE(result) TYPE i.
  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_085 IMPLEMENTATION.
  METHOD add.
    DATA: lv_sum TYPE i VALUE 0.

    LOOP AT lst INTO DATA(lv_value) WHERE table_line IS NOT INITIAL.
      DATA(lv_index) = sy-tabix.
      IF lv_index MOD 2 = 0 AND lv_value MOD 2 = 0.
        lv_sum = lv_sum + lv_value.
      ENDIF.
    ENDLOOP.

    result = lv_sum.
  ENDMETHOD.
ENDCLASS.