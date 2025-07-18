CLASS z_humaneval_009 DEFINITION
  PUBLIC
  FINAL
  ABSTRACT.

  PUBLIC SECTION.
    CLASS-METHODS rolling_max
      IMPORTING
        VALUE(numbers) TYPE int4_table
      RETURNING
        VALUE(result)  TYPE int4_table.

ENDCLASS.

CLASS z_humaneval_009 IMPLEMENTATION.

  METHOD rolling_max.
    DATA: lv_running_max TYPE i.

    LOOP AT numbers INTO DATA(lv_number).
      IF sy-tabix = 1.
        lv_running_max = lv_number.
      ELSE.
        IF lv_number > lv_running_max.
          lv_running_max = lv_number.
        ENDIF.
      ENDIF.
      APPEND lv_running_max TO result.
    ENDLOOP.
  ENDMETHOD.

ENDCLASS.