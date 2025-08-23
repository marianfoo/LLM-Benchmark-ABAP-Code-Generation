CLASS ltcl_delivery_date_test DEFINITION FINAL FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      setup,
      teardown,

      " Test scenarios
      test_material_found_no_hol FOR TESTING,
      test_material_found_with_hol FOR TESTING,
      test_material_not_found FOR TESTING,
      test_weekend_skip FOR TESTING,
      test_multiple_holidays FOR TESTING,
      test_holiday_on_weekend FOR TESTING,
      test_zero_lead_time FOR TESTING,
      test_order_on_friday FOR TESTING,
      test_long_lead_time FOR TESTING.

ENDCLASS.

CLASS ltcl_delivery_date_test IMPLEMENTATION.

  METHOD setup.
    " Clean up any existing test data
    DELETE FROM zmaterial.
    DELETE FROM zholiday.

    " Insert test data for materials
    INSERT zmaterial FROM TABLE @( VALUE #(
      ( mat_id = 'TEST001' lead_time_days = 3 )
      ( mat_id = 'TEST002' lead_time_days = 1 )
      ( mat_id = 'TEST003' lead_time_days = 5 )
      ( mat_id = 'TEST004' lead_time_days = 0 )
      ( mat_id = 'TEST005' lead_time_days = 10 )
    ) ).

    " Insert test data for holidays
    INSERT zholiday FROM TABLE @( VALUE #(
      ( holiday_date = '20241224' name = 'Christmas Eve' )
      ( holiday_date = '20241225' name = 'Christmas Day' )
      ( holiday_date = '20241226' name = 'Boxing Day' )
      ( holiday_date = '20241231' name = 'New Year Eve' )
      ( holiday_date = '20250101' name = 'New Year Day' )
      ( holiday_date = '20241222' name = 'Holiday on Sunday' )  " Sunday holiday
    ) ).

    COMMIT WORK.

  ENDMETHOD.

  METHOD teardown.
    " Clean up test data
    DELETE FROM zmaterial.
    DELETE FROM zholiday.
    COMMIT WORK.
  ENDMETHOD.

  METHOD test_material_found_no_hol.
    " Test case: Material found, no holidays in range
    " Order on Monday 2024-12-16, 3 working days = Thursday 2024-12-19

    DATA(lv_result) = z_humaneval_erp_007=>calculate_delivery_date(
      order_date = '20241216'
      mat_id     = 'TEST001' ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = '20241219'
      msg = 'Should deliver on Thursday after 3 working days' ).

  ENDMETHOD.

  METHOD test_material_found_with_hol.
    " Test case: Material found, holidays in range
    " Order on Monday 2024-12-23, 3 working days, skip Christmas holidays
    " Expected: Friday 2024-12-27 (skip 24th, 25th, 26th holidays + weekend)

    DATA(lv_result) = z_humaneval_erp_007=>calculate_delivery_date(
      order_date = '20241223'
      mat_id     = 'TEST001' ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = '20250102'
      msg = 'Should skip Christmas holidays and deliver on Friday' ).

  ENDMETHOD.

  METHOD test_material_not_found.
    " Test case: Material not found, should return order date

    DATA(lv_result) = z_humaneval_erp_007=>calculate_delivery_date(
      order_date = '20241216'
      mat_id     = 'NONEXISTENT' ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = '20241216'
      msg = 'Should return order date when material not found' ).

  ENDMETHOD.

  METHOD test_weekend_skip.
    " Test case: Order on Friday, should skip weekend
    " Order on Friday 2024-12-13, 1 working day = Monday 2024-12-16

    DATA(lv_result) = z_humaneval_erp_007=>calculate_delivery_date(
      order_date = '20241213'
      mat_id     = 'TEST002' ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = '20241216'
      msg = 'Should skip weekend and deliver on Monday' ).

  ENDMETHOD.

  METHOD test_multiple_holidays.
    " Test case: Multiple consecutive holidays
    " Order on Friday 2024-12-27, 3 working days, skip New Year holidays
    " Expected: Skip weekend + New Year holidays

    DATA(lv_result) = z_humaneval_erp_007=>calculate_delivery_date(
      order_date = '20241227'
      mat_id     = 'TEST001' ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = '20250103'
      msg = 'Should skip multiple holidays and weekend' ).

  ENDMETHOD.

  METHOD test_holiday_on_weekend.
    " Test case: Holiday falls on weekend (should not affect calculation)
    " Holiday on Sunday 2024-12-22, order on Friday 2024-12-20
    " 1 working day = Monday 2024-12-23

    DATA(lv_result) = z_humaneval_erp_007=>calculate_delivery_date(
      order_date = '20241220'
      mat_id     = 'TEST002' ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = '20241223'
      msg = 'Holiday on weekend should not affect working day calculation' ).

  ENDMETHOD.

  METHOD test_zero_lead_time.
    " Test case: Material with zero lead time

    DATA(lv_result) = z_humaneval_erp_007=>calculate_delivery_date(
      order_date = '20241216'
      mat_id     = 'TEST004' ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = '20241216'
      msg = 'Should return order date for zero lead time' ).

  ENDMETHOD.

  METHOD test_order_on_friday.
    " Test case: Order on Friday with 5 working days
    " Order on Friday 2024-12-13, 5 working days = Friday 2024-12-20

    DATA(lv_result) = z_humaneval_erp_007=>calculate_delivery_date(
      order_date = '20241213'
      mat_id     = 'TEST003' ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = '20241220'
      msg = 'Should count working days correctly over weekends' ).

  ENDMETHOD.


  METHOD test_long_lead_time.
    " Test case: Long lead time spanning multiple weeks
    " Order on Monday 2024-12-16, 10 working days = Monday 2024-12-30

    DATA(lv_result) = z_humaneval_erp_007=>calculate_delivery_date(
      order_date = '20241216'
      mat_id     = 'TEST005' ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = '20250106'
      msg = 'Should handle long lead times correctly' ).

  ENDMETHOD.

ENDCLASS.